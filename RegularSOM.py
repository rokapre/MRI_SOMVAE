import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RegularSOM:
    def __init__(self, num_embeddings = 256, embedding_dim = 32, som_h = 16, som_w = 16, alpha = 6,beta = 1):
        assert som_h*som_w == num_embeddings
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._som_h = som_h
        self._som_w = som_w
        #self.eta = eta #learning rate 
        self.alpha = alpha #only used for loss tracking
        self.beta = beta #only used for loss tracking
        #self.lam = lam # inverse of time constant for learning rate decay 
        self.idx_to_coord = {}  # needed to convert row index on Embeddings to grid coordinate
        idx = 0
        for i in range(som_h):
            for j in range(som_w):
                self.idx_to_coord[idx] = (i, j)
                idx += 1
        # converts grid coordinate to row index  of Embeddings
        self.coord_to_idx = {v: k for k, v in self.idx_to_coord.items()}
        self._embedding = nn.Embedding(num_embeddings, embedding_dim)
        self._embedding.weight.data.normal_()
        self.total_train_iters = 0

    def get_neighbors(self, curr_idx):
        """
        This method gets the neighbors 1 of the current node in the SOM grid 

        """
        curr_h, curr_w = self.idx_to_coord[curr_idx]
        top_idx = self.coord_to_idx[(
            curr_h - 1, curr_w)] if curr_h - 1 >= 0 else ""
        bottom_idx = self.coord_to_idx[(
            curr_h + 1, curr_w)] if curr_h + 1 < self._som_h else ""
        left_idx = self.coord_to_idx[(
            curr_h, curr_w - 1)] if curr_w - 1 >= 0 else ""
        right_idx = self.coord_to_idx[(
            curr_h, curr_w + 1)] if curr_w + 1 < self._som_w else ""
        neighbors = [curr_idx, top_idx, bottom_idx, left_idx, right_idx]
        neighbors = list(filter(lambda x: x != "", neighbors))
        return(neighbors)

    def train_step(self, inputs,eta=1):
  
        with torch.no_grad():
            inputs = inputs.permute(0, 2, 3, 4, 1).contiguous()
            input_shape = inputs.shape
            flat_inputs = inputs.view(-1,self._embedding_dim)

            dist = (torch.sum(flat_inputs.detach()**2, dim=1, keepdims=True) +
                    torch.sum(self._embedding.weight**2, dim=1)) - 2*torch.matmul(flat_inputs.detach(), self._embedding.weight.t())
            encoding_idx = torch.argmin(dist, dim=1, keepdims=True)
            encodings = torch.zeros(encoding_idx.shape[0],self._num_embeddings)#, device = inputs.device)
            encodings.scatter_(1,encoding_idx,1)

            quantized = torch.matmul(encodings,self._embedding.weight).view(input_shape) #Just for Loss calculation purposes

            commitment_loss = F.mse_loss(quantized,inputs)

            indicator = torch.zeros(dist.shape, device = inputs.device)
            n_neighbors = torch.zeros(indicator.shape[0],requires_grad=False, device = inputs.device)
            for i in range(flat_inputs.shape[0]):
                curr_idx = encoding_idx[i].item()
                curr_node = self._embedding.weight[curr_idx]

                neighbor_idx = self.get_neighbors(curr_idx)
                neighbor_nodes = self._embedding.weight[neighbor_idx] #includes best matching (current) node
                n_neighbors[i] = len(neighbor_idx)
                indicator[i,neighbor_idx] = 1

                curr_to_neighbor_dist = (curr_node-neighbor_nodes).pow(2).sum(dim=1,keepdims=True) #distance squared from current to neighbor nodes
                nu = torch.exp(-curr_to_neighbor_dist/2)
                sample_to_nodes = flat_inputs[i,:] - neighbor_nodes
                update = eta*nu*sample_to_nodes
                self._embedding.weight[neighbor_idx] = self._embedding.weight[neighbor_idx] + update
                
            total_neighbors = n_neighbors.sum()
            newdist = torch.multiply(dist,indicator)
            somloss = newdist.sum().divide(total_neighbors)
            loss = self.alpha*commitment_loss + self.beta*somloss
 
            #self.eta *= np.exp(-1/self.lam)
            #print(f"Epoch {epoch} complete. Training Loss: {loss[epoch]}, Commitment Loss: {commitment_losses[i]}, SOM Loss: {som_losses[i]}")
            
        self.total_train_iters += 1
        return(loss)


def encoder_output(AEmodel,dataloader):
    encoded = []
    for Xin,Xout in dataloader:
        encoded.append(AEmodel.encoder(Xin))
    return(encoded)



def train_SOM(encoded_data,SOM,epochs,eta_0=1,lam = 1000):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    eta = eta_0
    losses = np.zeros(epochs)
    loss_per_batch = []
    for epoch in range(epochs):
        for Xenc in encoded_data:
            Xenc.to(device)
            loss_per_batch.append(SOM.train_step(Xenc,eta=eta))
        losses[epoch] = np.mean(np.array(loss_per_batch))
        loss_per_batch = []
        print(f"Epoch {epoch} Complete. Average Training Loss: {losses[epoch]} ")
        eta *= np.exp(-1/lam)
    return(losses)








