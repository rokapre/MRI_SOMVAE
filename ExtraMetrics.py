import numpy as np
import torch.nn.functional as F 
import torch 


def make_uniform3D_window(length,width=None,height=None):
    """
    This creates the Uniform Kernel for the SSIM calculation
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    if width==None:
        width = length
    if height == None: 
        height = length
    volume = length*width*height
    window = 1/volume * torch.ones((length,width,height))
    return(window.unsqueeze(0).unsqueeze(0).to(device))

#window = make_uniform3D_window(4)
#image_practice[0,0,1,0,0]= 1
#F.conv3d(a,window)

def SSIM(Xorig,Xrecon,length=9,width=None,height=None):
    """
    Structural Similarity Index Measure (SSIM)

    Calculates the SSIM between 2 images given the window size. A 
    uniform kernel is used for simplicity. 

    Note: This calculation can be a bottleneck for training time if used 
    in the training loop. 

    """
    window = make_uniform3D_window(length=length,width=width,height=height)
    mu_orig = F.conv3d(Xorig,window)
    mu_recon = F.conv3d(Xrecon,window)

    mu_orig_sq = mu_orig.pow(2)
    mu_recon_sq = mu_recon.pow(2)
    mu_orig_mu_recon = mu_orig*mu_recon

    var_orig = F.conv3d(Xorig.pow(2),window) - mu_orig_sq
    var_recon = F.conv3d(Xrecon.pow(2),window) - mu_recon_sq

    cov_origrecon = F.conv3d(Xorig*Xrecon,window) - mu_orig_mu_recon

    Imax = torch.amax(Xorig,dim=(1,2,3,4))
    Imin = torch.amin(Xorig,dim=(1,2,3,4))

    L = (Imax - Imin).reshape(mu_orig_sq.shape[0],1,1,1,1)

    c1 = (0.01*L).pow(2)
    c2 = (0.03*L).pow(2)

    numerator = (2*mu_orig_mu_recon+c1)*(2*cov_origrecon + c2)
    denominator = (mu_orig_sq+mu_recon_sq+c1)*(var_orig+var_recon+c2)

    ssim = (numerator/denominator).mean(dim=(1,2,3,4))
    return(ssim)

def PSNR(Xorig,Xrecon):
    """
    Calculates Peak Signal to Noise Ratio (PSNR) in dB between an image and its reconstruction
    """
    mse = (Xorig-Xrecon).pow(2).mean(dim=(1,2,3,4))
    maxI = Xorig.amax(dim=(1,2,3,4))

    psnr = 20*torch.log10(maxI) - 10*torch.log10(mse)

    return(psnr)

def MoranScore(SOMmodel):
    codebook = SOMmodel.quantization._embedding.weight.detach().numpy()
    som_h = SOMmodel.quantization._som_h
    som_w = SOMmodel.quantization._som_w
    idx_to_coord = SOMmodel.quantization.idx_to_coord

    N = codebook.shape[0]
    #codebook = SS.fit_transform(codebook)
    #codebook = codebook.sum(axis=1,keepdims=True)
    codebook = codebook[:,10].reshape(N,1)
    codebook_mean = codebook.mean(axis=0,keepdims=True)

    centered_codebook = codebook - codebook_mean

    XXT = np.matmul(centered_codebook,centered_codebook.T)

    W_ij = np.zeros(XXT.shape) #weight matrix

    for i in range(codebook.shape[0]):
        for j in range(codebook.shape[0]):
            hi,wi = idx_to_coord[i] #hi and wj here are coordinates of SOM grid
            hj,wj = idx_to_coord[j]
            som_dist = np.abs(hj-hi) + np.abs(wj-wi)
            W_ij[i,j] = np.exp(-som_dist)

    XXT_diag = np.diag(XXT)
    W = np.sum(W_ij)

    I_Moran = N/W * (np.sum(W_ij * XXT))/(np.sum(XXT_diag))

    return(I_Moran)

