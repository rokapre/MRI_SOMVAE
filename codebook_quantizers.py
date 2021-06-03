import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class VectorQuantizerEMA(nn.Module):
    """
    Uses exponential moving average (EMA) update to update the codebook. 

    Original VQ layer is from Oord et al. 2017 https://colab.research.google.com/github/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb. 
   
    """
    def __init__(self, num_embeddings = 256, embedding_dim = 32, commitment_cost = 6, decay = 0.99, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost
        
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()
        
        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        # convert inputs from BCHWD -> BHWDC
        inputs = inputs.permute(0, 2, 3, 4, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)
            
            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon) * n)
            
            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)
            
            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss
        
        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # convert quantized from BHWDC -> BCHWD
        return loss, quantized.permute(0, 4, 1, 2, 3).contiguous(), perplexity, encodings



class SOMQuantizer(nn.Module):
    """
    This is the main modification we propose to the VQ codebook layer. Instead of using vector quantization,
    the SOM layer is better able to capture the topology of the data. Rather than having very disjoint
    clusters, the SOM nodes form a grid. Each voxel of the incoming image is assigned to the closest node 
    based on Euclidean distance. Gradient based optimization is used to update this node AND the ones 
    immediately around it in the SOM grid. This results in a more connected set of clusters which preserves
    more information. 

    The layer is a combination of the VQ layer and the algorithm described in 
    Fortuin et al 2019 https://arxiv.org/pdf/1806.02199.pdf

    alpha is the VQ commitment loss multiplier while beta is the SOM loss multiplier

    """
    def __init__(self,num_embeddings = 256,embedding_dim = 32,som_h = 16,som_w = 16,alpha = 6,beta=1,geometry = "rectangular"):
        super(SOMQuantizer,self).__init__()
        assert som_h*som_w == num_embeddings
        assert geometry in ["rectangular","toroid"]
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._som_h = som_h
        self._som_w = som_w 
        self.alpha = alpha
        self.beta = beta 
        self._embedding = nn.Embedding(num_embeddings,embedding_dim)
        self.idx_to_coord = {} #needed to convert row index on Embeddings to grid coordinate
        idx = 0 
        for i in range(som_h):
            for j in range(som_w):
                self.idx_to_coord[idx] = (i,j)
                idx+=1
        self.coord_to_idx = {v:k for k,v in self.idx_to_coord.items()} #converts grid coordinate to row index  of Embeddings
        self.geometry = geometry 
    
    def get_neighbors(self,curr_idx):
        """
        This method gets the neighbors 1 of the current node in the SOM grid 

        """
        curr_h,curr_w = self.idx_to_coord[curr_idx]
        if self.geometry == "rectangular":
            top_idx = self.coord_to_idx[(curr_h - 1,curr_w)] if curr_h - 1 >= 0 else ""
            bottom_idx = self.coord_to_idx[(curr_h + 1,curr_w)] if curr_h + 1 < self._som_h else ""
            left_idx = self.coord_to_idx[(curr_h,curr_w - 1)] if curr_w - 1 >= 0 else ""
            right_idx = self.coord_to_idx[(curr_h,curr_w + 1)] if curr_w + 1 < self._som_w else ""
        elif self.geometry == "toroid":
            top_idx = self.coord_to_idx[(curr_h - 1,curr_w)] if curr_h - 1 >= 0 else self.coord_to_idx[(self._som_h - 1,curr_w)]
            bottom_idx = self.coord_to_idx[(curr_h + 1,curr_w)] if curr_h + 1 < self._som_h else 0
            left_idx = self.coord_to_idx[(curr_h,curr_w - 1)] if curr_w - 1 >= 0 else self.coord_to_idx[(curr_h,(self._som_w - 1))]
            right_idx = self.coord_to_idx[(curr_h,curr_w + 1)] if curr_w + 1 < self._som_w else 0
        neighbors = [curr_idx,top_idx,bottom_idx,left_idx,right_idx]
        neighbors = list(filter(lambda x: x!= "", neighbors))
        return(neighbors)

        
    def forward(self,inputs):

        inputs = inputs.permute(0, 2, 3, 4, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_inputs = inputs.view(-1, self._embedding_dim)
        
        dist = (torch.sum(flat_inputs.detach()**2, dim=1,keepdims=True) + 
                torch.sum(self._embedding.weight**2, dim=1)) - 2*torch.matmul(flat_inputs.detach(),self._embedding.weight.t())
        encoding_idx = torch.argmin(dist,dim=1,keepdims=True)
        encodings = torch.zeros(encoding_idx.shape[0],self._num_embeddings, device = inputs.device)
        encodings.scatter_(1,encoding_idx,1)

        #This part is the same as VQVAE 
        quantized = torch.matmul(encodings,self._embedding.weight).view(input_shape)

        commitment_loss = F.mse_loss(quantized,inputs)

        indicator = torch.zeros(dist.shape, device = inputs.device)
        n_neighbors = torch.zeros(indicator.shape[0],requires_grad=False, device = inputs.device)
        for i in range(indicator.shape[0]):
            curr_idx = encoding_idx[i].item()
            neighbor_idx = self.get_neighbors(curr_idx)
            n_neighbors[i] = len(neighbor_idx)
            indicator[i,neighbor_idx] = 1 

        total_neighbors = n_neighbors.sum()
        newdist = torch.multiply(dist,indicator)
        somloss = newdist.sum().divide(total_neighbors)

        loss = self.alpha*commitment_loss + self.beta*somloss
        quantized = inputs + (quantized - inputs).detach()
        #avg_probs = torch.mean(encodings, dim = 0)
        #perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        return(loss, quantized.permute(0, 4, 1, 2, 3).contiguous(), encodings)


class PSOMQuantizer(nn.Module):
    """
    The PSOMQuantizer is a probabilistic modification to the SOMQuantizer. The SOM assigns
    each incoming voxel to just 1 node and updates the neighbors. The PSOM instead assigns a probability
    distribution over the nodes, so each voxel "belongs" to multiple nodes. A combination of a 
    cluster assignment hardening (CAH) loss and a probabilistic SOM loss is used. sij below represents the 
    probability of voxel i belonging to cluster j. 

    Details for the PSOM can be found in the following paper: https://dl.acm.org/doi/pdf/10.1145/3450439.3451872
    """
    def __init__(self,num_embeddings = 256,embedding_dim = 32,som_h = 16,som_w = 16,gamma = 1,beta = 2,geometry = "rectangular", df = 10,Kappa=2):
        super(PSOMQuantizer,self).__init__()
        assert som_h*som_w == num_embeddings
        assert geometry in ["rectangular","toroid"]
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._som_h = som_h
        self._som_w = som_w 
        self.gamma = gamma 
        self.beta = beta 
        self._embedding = nn.Embedding(num_embeddings,embedding_dim)
        self._embedding.weight.data.normal_()
        self.idx_to_coord = {} #needed to convert row index on Embeddings to grid coordinate
        idx = 0 
        for i in range(som_h):
            for j in range(som_w):
                self.idx_to_coord[idx] = (i,j)
                idx+=1
        self.coord_to_idx = {v:k for k,v in self.idx_to_coord.items()} #converts grid coordinate to row index  of Embeddings
        self.geometry = geometry
        self.df = df
        self.Kappa = Kappa 
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if use_cuda else "cpu")
        self.adjacency = torch.zeros((num_embeddings,num_embeddings),device=self.device)
        for i in range(num_embeddings):
            self.adjacency[i,self.get_neighbors(i)] = 1

    def get_neighbors(self,curr_idx):
        """
        This method gets the neighbors 1 of the current node in the SOM grid 

        """
        curr_h,curr_w = self.idx_to_coord[curr_idx]
        if self.geometry == "rectangular":
            top_idx = self.coord_to_idx[(curr_h - 1,curr_w)] if curr_h - 1 >= 0 else ""
            bottom_idx = self.coord_to_idx[(curr_h + 1,curr_w)] if curr_h + 1 < self._som_h else ""
            left_idx = self.coord_to_idx[(curr_h,curr_w - 1)] if curr_w - 1 >= 0 else ""
            right_idx = self.coord_to_idx[(curr_h,curr_w + 1)] if curr_w + 1 < self._som_w else ""
        elif self.geometry == "toroid":
            top_idx = self.coord_to_idx[(curr_h - 1,curr_w)] if curr_h - 1 >= 0 else self.coord_to_idx[(self._som_h - 1,curr_w)]
            bottom_idx = self.coord_to_idx[(curr_h + 1,curr_w)] if curr_h + 1 < self._som_h else 0
            left_idx = self.coord_to_idx[(curr_h,curr_w - 1)] if curr_w - 1 >= 0 else self.coord_to_idx[(curr_h,(self._som_w - 1))]
            right_idx = self.coord_to_idx[(curr_h,curr_w + 1)] if curr_w + 1 < self._som_w else 0
        neighbors = [top_idx,bottom_idx,left_idx,right_idx]
        neighbors = list(filter(lambda x: x!= "", neighbors))
        return(neighbors)

        
    def forward(self,inputs):

        inputs = inputs.permute(0, 2, 3, 4, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_inputs = inputs.view(-1, self._embedding_dim)
        
        dist = (torch.sum(flat_inputs.detach()**2, dim=1,keepdims=True) + 
                torch.sum(self._embedding.weight**2, dim=1)) - 2*torch.matmul(flat_inputs.detach(),self._embedding.weight.t())
        
        sij_numer = (1 + dist/self.df).pow(-0.5*(self.df+1))
        sij_denom = sij_numer.sum(dim=1,keepdims=True)

        sij = torch.divide(sij_numer,sij_denom)

        sij_Kappa = sij.pow(self.Kappa)

        sumi_sij = sij.sum(dim=0,keepdims=True)

        tij_numer = torch.divide(sij_Kappa,sumi_sij)
        tij_denom = tij_numer.sum(dim=1,keepdims=True)

        tij = torch.divide(tij_numer,tij_denom)

        log_sij = torch.log(sij)
        log_tij = torch.log(tij)

        CAH_loss = torch.sum(torch.multiply(tij,log_tij-log_sij))

        log_sie = torch.matmul(log_sij,self.adjacency)

        sumj_sij_log_sie = torch.sum(torch.multiply(sij,log_sie), dim = 1)

        SOM_loss = -sumj_sij_log_sie.mean()
        total_loss = self.gamma*CAH_loss + self.beta*SOM_loss

        return({"CAH_loss": CAH_loss,"SOM_loss": SOM_loss,"vq_loss": total_loss})
