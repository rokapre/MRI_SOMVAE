import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class VQEncoder(nn.Module):
    """
    Encoder architecture that outputs a volume to a quantization layer. Also outputs skip connections

    The encoder architecture is taken from https://github.com/RdoubleA/DWI-inpainting
    """
    def __init__(self, num_channels, num_filters = 8, embedding_dim = 32, skip_connections = False, batchnorm = False):
        super().__init__()

        self.skip = skip_connections

        if batchnorm:
            self.conv1 = nn.Sequential(
                nn.Conv3d(num_channels, num_filters, kernel_size = 4, stride = 2, padding = 1),
                nn.BatchNorm3d(num_filters),
                nn.ReLU()
            )

            self.conv2 = nn.Sequential(
                nn.Conv3d(num_filters, num_filters * 2, kernel_size = 4, stride = 2, padding = 1),
                nn.BatchNorm3d(num_filters * 2),
                nn.ReLU()
            )

            self.conv3 = nn.Sequential(
                nn.Conv3d(num_filters * 2, num_filters * 4, kernel_size = 4, stride = 2, padding = 1),
                nn.BatchNorm3d(num_filters * 4),
                nn.ReLU()
            )

        else:

            self.conv1 = nn.Sequential(
                nn.Conv3d(num_channels, num_filters, kernel_size = 4, stride = 2, padding = 1),
                nn.ReLU()
            )

            self.conv2 = nn.Sequential(
                nn.Conv3d(num_filters, num_filters * 2, kernel_size = 4, stride = 2, padding = 1),
                nn.ReLU()
            )

            self.conv3 = nn.Sequential(
                nn.Conv3d(num_filters * 2, num_filters * 4, kernel_size = 4, stride = 2, padding = 1),
                nn.ReLU()
            )

        self.conv4 = nn.Sequential(
            nn.Conv3d(num_filters * 4, embedding_dim, kernel_size = 4, stride = 2, padding = 1)
        )


    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        ze = self.conv4(x3)

        if self.skip:
            return x1, x2, x3, ze
        else:
            return ze



class VQDecoder(nn.Module):
    """
    Decoder architecture that accepts a volume from a quantization layer

    For use without skip connections 

    The decoder architecture is taken from https://github.com/RdoubleA/DWI-inpainting
    """
    def __init__(self, num_channels, num_filters = 8, embedding_dim = 32, batchnorm = False):
        super().__init__()

        if batchnorm:
            self.conv1 = nn.Sequential(
                nn.ConvTranspose3d(embedding_dim, num_filters * 4, kernel_size = 4, stride = 2, padding = 1),
                nn.BatchNorm3d(num_filters * 4),
                nn.ReLU()
                )

            self.conv2 = nn.Sequential(
                nn.ConvTranspose3d(num_filters * 4, num_filters * 2, kernel_size = 4, stride = 2, padding = 1),
                nn.BatchNorm3d(num_filters * 2),
                nn.ReLU()
                )


            self.conv3 = nn.Sequential(
                nn.ConvTranspose3d(num_filters * 2, num_filters, kernel_size = 4, stride = 2, padding = 1),
                nn.BatchNorm3d(num_filters),
                nn.ReLU()
                )

        else:
            self.conv1 = nn.Sequential(
                nn.ConvTranspose3d(embedding_dim, num_filters * 4, kernel_size = 4, stride = 2, padding = 1),
                nn.ReLU()
                )

            self.conv2 = nn.Sequential(
                nn.ConvTranspose3d(num_filters * 4, num_filters * 2, kernel_size = 4, stride = 2, padding = 1),
                nn.ReLU()
                )


            self.conv3 = nn.Sequential(
                nn.ConvTranspose3d(num_filters * 2, num_filters, kernel_size = 4, stride = 2, padding = 1),
                nn.ReLU()
                )

        self.conv4 = nn.ConvTranspose3d(num_filters, num_channels, kernel_size = 4, stride = 2, padding = 1)
        

    def forward(self, zq):

        x1 = self.conv1(zq)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x_recon = self.conv4(x3)

        return x_recon



class VQDecoder_skip(nn.Module):
    """
    Decoder architecture that accepts a volume from a quantization layer and skip connections from the encoder

    For use with skip connections (which we found does not tend to work well for learning embeddings)

    The decoder architecture is taken from https://github.com/RdoubleA/DWI-inpainting
  
    """
    def __init__(self, num_channels, num_filters = 8, embedding_dim = 32, batchnorm = False):
        super().__init__()

        if batchnorm:
            self.conv1 = nn.Sequential(
                nn.ConvTranspose3d(embedding_dim, num_filters * 4, kernel_size = 4, stride = 2, padding = 1),
                nn.BatchNorm3d(num_filters * 4),
                nn.ReLU()
                )

            self.conv2 = nn.Sequential(
                nn.ConvTranspose3d(num_filters * 8, num_filters * 2, kernel_size = 4, stride = 2, padding = 1),
                nn.BatchNorm3d(num_filters * 2),
                nn.ReLU()
                )


            self.conv3 = nn.Sequential(
                nn.ConvTranspose3d(num_filters * 4, num_filters, kernel_size = 4, stride = 2, padding = 1),
                nn.BatchNorm3d(num_filters),
                nn.ReLU()
                )

        else:
            self.conv1 = nn.Sequential(
                nn.ConvTranspose3d(embedding_dim, num_filters * 4, kernel_size = 4, stride = 2, padding = 1),
                nn.ReLU()
                )

            self.conv2 = nn.Sequential(
                nn.ConvTranspose3d(num_filters * 8, num_filters * 2, kernel_size = 4, stride = 2, padding = 1),
                nn.ReLU()
                )


            self.conv3 = nn.Sequential(
                nn.ConvTranspose3d(num_filters * 4, num_filters, kernel_size = 4, stride = 2, padding = 1),
                nn.ReLU()
                )

        self.conv4 = nn.ConvTranspose3d(num_filters * 2, num_channels, kernel_size = 4, stride = 2, padding = 1)
        

    def forward(self, zq, encoder_layer1_output, encoder_layer2_output, encoder_layer3_output):

        x1 = self.conv1(zq)
        x2 = torch.cat([encoder_layer3_output, x1], dim = 1)
        x3 = self.conv2(x2)
        x4 = torch.cat([encoder_layer2_output, x3], dim = 1)
        x5 = self.conv3(x4)
        x6 = torch.cat([encoder_layer1_output, x5], dim = 1)
        x_recon = self.conv4(x6)

        return x_recon


