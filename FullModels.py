import torch
import torch.nn as nn
from codebook_quantizers import *
from encoder_decoder import *


class VanillaAE(nn.Module):
    """
    Vanilla Autoencoder Model 

    This model is a regular autoencoder model which assigns a vector of length 
    embedding_dim to each voxel from the incoming image. 

    num_channels = number of input channels. In our case, this is just 1 due to grayscale images
    num_filters = number of filters in the first convolutional layer, this doubles with every convolutional layer
    embedding_dim = dimensionality of vectors in the embedding space
    batchnorm = if True, include batchnorm layers after every convolutional layer

    """
    def __init__(self, num_channels, num_filters, embedding_dim = 32, batchnorm = False):
        super().__init__()
        self.encoder = VQEncoder(num_channels, num_filters, embedding_dim, skip_connections = False, batchnorm = batchnorm)
        self.decoder = VQDecoder(num_channels, num_filters, embedding_dim, batchnorm)

    def forward(self, x):
        ze = self.encoder(x)
        x_recon = self.decoder(ze)

        outputs = {'x_out': x_recon}

        return outputs


class VanillaVAE(nn.Module):
    """
    Vanilla Variational Autoencoder Model 

    This model is a regular variational autoencoder model which samples at the voxel level

    num_channels = number of input channels. In our case, this is just 1 due to grayscale images
    num_filters = number of filters in the first convolutional layer, this doubles with every convolutional layer
    embedding_dim = dimensionality of vectors in the embedding space
    batchnorm = if True, include batchnorm layers after every convolutional layer

    """
    def __init__(self, num_channels, num_filters, embedding_dim = 32, batchnorm = False):
        assert embedding_dim % 2 == 0
        super().__init__()
        self.encoder_mu = VQEncoder(num_channels, num_filters, embedding_dim, skip_connections = False, batchnorm = batchnorm)
        self.encoder_sigma = VQEncoder(num_channels, num_filters, embedding_dim, skip_connections = False, batchnorm = batchnorm)
        self.decoder = VQDecoder(num_channels, num_filters, embedding_dim, batchnorm)
        self.embedding_dim = embedding_dim 

    def reparametrize(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        new_sample = mu + (eps*std)
        return(new_sample)



    def forward(self, x):
        ze_mu = self.encoder_mu(x)
        ze_sigma = self.encoder_sigma(x)

        ze_mu = ze_mu.permute(0, 2, 3, 4, 1).contiguous()
        ze_sigma = ze_sigma.permute(0, 2, 3, 4, 1).contiguous()
        ze_permuted_shape = ze_mu.shape

        
        ze_mu_flat = ze_mu.view(-1,self.embedding_dim)
        ze_log_var_flat = ze_sigma.view(-1,self.embedding_dim)

        ze_new = self.reparametrize(ze_mu_flat,ze_log_var_flat)
        ze_new = ze_new.reshape(ze_permuted_shape).permute(0,4,1,2,3)

        KLD = (ze_log_var_flat.exp() + ze_mu_flat.pow(2) - 0.5*ze_log_var_flat - 1/2).mean()

        x_recon = self.decoder(ze_new)

        outputs = {'x_out': x_recon,"KLD_loss": KLD}

        return outputs


class VQVAE3D(nn.Module):
    """
    VQVAE and U-VQVAE model

    Replace the latent layer in a variational autoencoder with vector quantization as described in Oord et al. 2017
    Has skip connection setting. The output is reconstructed only from the quantized representation
    This is the baseline architecture taken from https://github.com/RdoubleA/DWI-inpainting

    num_channels = number of input channels. In our case, this is just 1 due to grayscale images
    num_filters = number of filters in the first convolutional layer, this doubles with every convolutional layer
    embedding_dim = dimensionality of vectors in the embedding space
    num_embeddings = number of vectors in the embedding space
    skip_connections = if True, the model is U-VQVAE. if False, the model is VQVAE.
    batchnorm = if True, include batchnorm layers after every convolutional layer

    """
    def __init__(self, num_channels, num_filters, embedding_dim = 32, num_embeddings = 512, commitment_cost = 6, skip_connections = False, batchnorm = True):
        super().__init__()
        self.skip = skip_connections
        self.encoder = VQEncoder(num_channels, num_filters, embedding_dim, skip_connections = skip_connections, batchnorm = batchnorm)
        self.quantization = VectorQuantizerEMA(num_embeddings = num_embeddings, embedding_dim = embedding_dim, commitment_cost = commitment_cost)
        # If skip connections enabled, use a different class for decoder that uses skip connections
        if skip_connections:
            self.decoder = VQDecoder_skip(num_channels, num_filters, embedding_dim, batchnorm)
        else:
            self.decoder = VQDecoder(num_channels, num_filters, embedding_dim, batchnorm)

    def forward(self, x):
        if self.skip:
            x1, x2, x3, ze = self.encoder(x)
            loss, zq, perplexity, _ = self.quantization(ze)
            x_recon = self.decoder(zq, x1, x2, x3)
        else:
            ze = self.encoder(x)
            loss, zq, perplexity, _ = self.quantization(ze)
            x_recon = self.decoder(zq)

        outputs = {'x_out': x_recon,
                   'vq_loss': loss}

        return outputs



class NewVQVAE3D(nn.Module):
    """
    VQVAE model with 2 reconstruction outputs 

    Replace the latent layer in a variational autoencoder with vector quantization as described in Oord et al. 2017
    Changes the VQVAE3D architecture to reconstruct images from direct encoder output (ze) in addition to embeddings (zq).

    num_channels = number of input channels. In our case, this is just 1 due to grayscale images
    num_filters = number of filters in the first convolutional layer, this doubles with every convolutional layer
    embedding_dim = dimensionality of vectors in the embedding space
    num_embeddings = number of vectors in the embedding space
    batchnorm = if True, include batchnorm layers after every convolutional layer

    """
    def __init__(self, num_channels, num_filters, embedding_dim = 32, num_embeddings = 512, commitment_cost = 6, batchnorm = True):
        super().__init__()
        self.encoder = VQEncoder(num_channels, num_filters, embedding_dim, skip_connections = False, batchnorm = batchnorm)
        self.quantization = VectorQuantizerEMA(num_embeddings = num_embeddings, embedding_dim = embedding_dim, commitment_cost = commitment_cost)
        self.decoder = VQDecoder(num_channels, num_filters, embedding_dim, batchnorm)

    def forward(self, x):
        
        ze = self.encoder(x)
        loss, zq, perplexity, _ = self.quantization(ze)
        x_recon_zq = self.decoder(zq)
        x_recon_ze = self.decoder(ze)

        outputs = {'x_out_zq': x_recon_zq, 'x_out_ze': x_recon_ze, 'vq_loss': loss}

        return outputs

    
class SOMVAE3D(nn.Module):
    """
    SOMVAE model

    Replaces the latent layer in a variational autoencoder with a self organized map (SOM) as in 
    "SOM-VAE: INTERPRETABLE DISCRETE REPRESENTATION LEARNING ON TIME SERIES" 
    Reconstructs image from direct encoder output (ze) in addition to quantized representation (zq)

    num_channels = number of input channels. In our case, this is just 1 due to grayscale images
    num_filters = number of filters in the first convolutional layer, this doubles with every convolutional layer
    embedding_dim = dimensionality of vectors in the embedding space
    num_embeddings = number of vectors in the embedding space
    som_h = height of SOM map, som_h*som_w must be equal to num_embeddings
    som_w = width of SOM map 
    alpha = commitment cost multiplier 
    beta = SOM cost multiplier 
    batchnorm = if True, include batchnorm layers after every convolutional layer

    """
    def __init__(self, num_channels, num_filters, embedding_dim = 32, num_embeddings = 256, som_h = 16, som_w = 16, alpha = 6, beta = 1, geometry = "rectangular", batchnorm = True):
        super().__init__()
        self.encoder = VQEncoder(num_channels, num_filters, embedding_dim, skip_connections = False, batchnorm = batchnorm)
        self.quantization = SOMQuantizer(num_embeddings = num_embeddings, embedding_dim = embedding_dim, som_h = som_h, som_w = som_w, alpha = alpha, beta = beta, geometry = geometry)
        self.decoder = VQDecoder(num_channels, num_filters, embedding_dim, batchnorm)

    def forward(self, x):
        
        ze = self.encoder(x)
        loss, zq, _ = self.quantization(ze)
        x_recon_zq = self.decoder(zq)
        x_recon_ze = self.decoder(ze)

        outputs = {'x_out_zq': x_recon_zq, 'x_out_ze': x_recon_ze, 'vq_loss': loss}

        return outputs


class SOMVAEContinuous(nn.Module):
    """
    SOM VAE with a continuous N(0,I) latent space

    This model is a regular variational autoencoder model which samples at the voxel level combined with the
    SOMQuantizer.

    num_channels = number of input channels. In our case, this is just 1 due to grayscale images
    num_filters = number of filters in the first convolutional layer, this doubles with every convolutional layer
    embedding_dim = dimensionality of vectors in the embedding space
    num_embeddings = number of vectors in the embedding space
    som_h = height of SOM map, som_h*som_w must be equal to num_embeddings
    som_w = width of SOM map 
    alpha = commitment cost multiplier 
    beta = SOM cost multiplier 
    batchnorm = if True, include batchnorm layers after every convolutional layer
    """
    def __init__(self, num_channels, num_filters, embedding_dim = 32, num_embeddings = 256, som_h = 16, som_w = 16, alpha = 6, beta = 1, geometry = "rectangular", batchnorm = True):
        assert embedding_dim % 2 == 0
        super().__init__()
        self.encoder_mu = VQEncoder(num_channels, num_filters, embedding_dim, skip_connections = False, batchnorm = batchnorm)
        self.encoder_sigma = VQEncoder(num_channels, num_filters, embedding_dim, skip_connections = False, batchnorm = batchnorm)
        self.quantization = SOMQuantizer(num_embeddings = num_embeddings, embedding_dim = embedding_dim, som_h = som_h, som_w = som_w, alpha = alpha, beta = beta, geometry = geometry)
        self.decoder = VQDecoder(num_channels, num_filters, embedding_dim, batchnorm)
        self.embedding_dim = embedding_dim 

    def reparametrize(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        new_sample = mu + (eps*std)
        return(new_sample)


    def forward(self, x):
        ze_mu = self.encoder_mu(x)
        ze_sigma = self.encoder_sigma(x)

        ze_mu = ze_mu.permute(0, 2, 3, 4, 1).contiguous()
        ze_sigma = ze_sigma.permute(0, 2, 3, 4, 1).contiguous()
        ze_permuted_shape = ze_mu.shape

        
        ze_mu_flat = ze_mu.view(-1,self.embedding_dim)
        ze_log_var_flat = ze_sigma.view(-1,self.embedding_dim)

        ze_new = self.reparametrize(ze_mu_flat,ze_log_var_flat)
        ze_new = ze_new.reshape(ze_permuted_shape).permute(0,4,1,2,3)

        #Feed sampled ze into SOM quantizer layer 

        quantizer_loss, zq_new,_ = self.quantization(ze_new)

        #Compute usual KLD Loss
        KLD = (ze_log_var_flat.exp() + ze_mu_flat.pow(2) - 0.5*ze_log_var_flat - 1/2).mean()

        x_recon_ze = self.decoder(ze_new)
        x_recon_zq = self.decoder(zq_new)


        outputs = {"x_out_ze": x_recon_ze,"x_out_zq": x_recon_zq,
                "KLD_loss": KLD,"vq_loss": quantizer_loss}

        return outputs

class PSOMVAE3D(nn.Module):
    """
    PSOMVAE model

    Replaces the latent layer in a variational autoencoder with a probabilistic self organized map.
    Described in https://dl.acm.org/doi/pdf/10.1145/3450439.3451872

    Reconstructs image from direct encoder output (ze) only. No quantizer image output. 

    num_channels = number of input channels. In our case, this is just 1 due to grayscale images
    num_filters = number of filters in the first convolutional layer, this doubles with every convolutional layer
    embedding_dim = dimensionality of vectors in the embedding space
    num_embeddings = number of vectors in the embedding space
    som_h = height of SOM map, som_h*som_w must be equal to num_embeddings
    som_w = width of SOM map 
    gamma = CAH cost multiplier 
    beta = SOM cost multiplier 
    batchnorm = if True, include batchnorm layers after every convolutional layer

    """
    def __init__(self, num_channels, num_filters, embedding_dim = 32, num_embeddings = 256, som_h = 16, som_w = 16, gamma = 1, beta = 1, geometry = "rectangular", batchnorm = True):
        super().__init__()
        self.encoder = VQEncoder(num_channels, num_filters, embedding_dim, skip_connections = False, batchnorm = batchnorm)
        self.quantization = PSOMQuantizer(num_embeddings = num_embeddings, embedding_dim = embedding_dim, som_h = som_h, som_w = som_w, gamma = gamma, beta = beta, geometry = geometry)
        self.decoder = VQDecoder(num_channels, num_filters, embedding_dim, batchnorm)

    def forward(self, x):
        
        ze = self.encoder(x)
        quantization_losses = self.quantization(ze)
        vq_loss = quantization_losses["vq_loss"]
        x_recon_ze = self.decoder(ze)

        outputs = {'x_out': x_recon_ze, 'vq_loss': vq_loss}

        return outputs



class PSOMVAEContinuous(nn.Module):
    """
    PSOM VAE with a continuous N(0,I) latent space

    This model is a regular variational autoencoder model which samples at the voxel level combined with the
    PSOMQuantizer.

    num_channels = number of input channels. In our case, this is just 1 due to grayscale images
    num_filters = number of filters in the first convolutional layer, this doubles with every convolutional layer
    embedding_dim = dimensionality of vectors in the embedding space
    num_embeddings = number of vectors in the embedding space
    som_h = height of SOM map, som_h*som_w must be equal to num_embeddings
    som_w = width of SOM map 
    gamma = CAH cost multiplier 
    beta = SOM cost multiplier 
    batchnorm = if True, include batchnorm layers after every convolutional layer
    """
    def __init__(self, num_channels, num_filters, embedding_dim = 32, num_embeddings = 256, som_h = 16, som_w = 16, gamma = 1, beta = 1, geometry = "rectangular", batchnorm = True):
        assert embedding_dim % 2 == 0
        super().__init__()
        self.encoder_mu = VQEncoder(num_channels, num_filters, embedding_dim, skip_connections = False, batchnorm = batchnorm)
        self.encoder_sigma = VQEncoder(num_channels, num_filters, embedding_dim, skip_connections = False, batchnorm = batchnorm)
        self.quantization = PSOMQuantizer(num_embeddings = num_embeddings, embedding_dim = embedding_dim, som_h = som_h, som_w = som_w, gamma = gamma, beta = beta, geometry = geometry)
        self.decoder = VQDecoder(num_channels, num_filters, embedding_dim, batchnorm)
        self.embedding_dim = embedding_dim 

    def reparametrize(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        new_sample = mu + (eps*std)
        return(new_sample)


    def forward(self, x):
        ze_mu = self.encoder_mu(x)
        ze_sigma = self.encoder_sigma(x)

        ze_mu = ze_mu.permute(0, 2, 3, 4, 1).contiguous()
        ze_sigma = ze_sigma.permute(0, 2, 3, 4, 1).contiguous()
        ze_permuted_shape = ze_mu.shape

        
        ze_mu_flat = ze_mu.view(-1,self.embedding_dim)
        ze_log_var_flat = ze_sigma.view(-1,self.embedding_dim)

        ze_new = self.reparametrize(ze_mu_flat,ze_log_var_flat)
        ze_new = ze_new.reshape(ze_permuted_shape).permute(0,4,1,2,3)

        #Feed sampled ze into SOM quantizer layer 

        quantization_losses = self.quantization(ze_new)
        vq_loss = quantization_losses["vq_loss"]

        #Compute usual KLD Loss
        KLD = (ze_log_var_flat.exp() + ze_mu_flat.pow(2) - 0.5*ze_log_var_flat - 1/2).mean()

        x_recon_ze = self.decoder(ze_new)


        outputs = {"x_out": x_recon_ze,
                "KLD_loss": KLD,"vq_loss": vq_loss}

        return outputs
