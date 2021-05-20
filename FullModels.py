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
    skip_connections = if True, the model is U-VQVAE. if False, the model is VQVAE.
    som_h = height of SOM map, som_h*som_w must be equal to num_embeddings
    som_w = width of SOM map 
    alpha = commitment cost multiplier 
    beta = SOM cost multiplier 
    batchnorm = if True, include batchnorm layers after every convolutional layer

    """
    def __init__(self, num_channels, num_filters, embedding_dim = 32, num_embeddings = 256, som_h = 16, som_w = 16, alpha = 6, beta = 1, batchnorm = True):
        super().__init__()
        self.encoder = VQEncoder(num_channels, num_filters, embedding_dim, skip_connections = False, batchnorm = batchnorm)
        self.quantization = SOMQuantizer(num_embeddings = num_embeddings, embedding_dim = embedding_dim, som_h = som_h, som_w = som_w, alpha = alpha, beta = beta)
        self.decoder = VQDecoder(num_channels, num_filters, embedding_dim, batchnorm)

    def forward(self, x):
        
        ze = self.encoder(x)
        loss, zq, _ = self.quantization(ze)
        x_recon_zq = self.decoder(zq)
        x_recon_ze = self.decoder(ze)

        outputs = {'x_out_zq': x_recon_zq, 'x_out_ze': x_recon_ze, 'vq_loss': loss}

        return outputs
