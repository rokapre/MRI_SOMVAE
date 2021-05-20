import os 
import numpy as np 
import pandas as pd 
import nibabel as nib

import torch 
import torchio as tio 

import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.manifold import TSNE 



def show_slices(slices):
    """
    Plots a list of slices of images
    """
    fig, axes = plt.subplots(1,len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T,cmap="gray",origin="lower")

def get_before_and_after(model,dataset,idx):
    """
    Given model and dataset and index, plots returns the before and after reconstruction 
    as a 3D numpy array for plotting with show_slices()
    """
    data_before = dataset[idx][0].unsqueeze(1)
    data_after = model(data_before)["x_out"]
    before = data_before.squeeze(0).squeeze(0).detach().numpy()
    after = data_after.squeeze(0).squeeze(0).detach().numpy()
    return before, after 


def get_before_and_after_combo(model,dataset,idx):
    """
    This is the same as get_before_and_after() but used on the models which 
    have 2 outputs, this one will take the output from the encoder. 
    """
    data_before = dataset[idx][0].unsqueeze(1)
    data_after = model(data_before)["x_out_ze"]
    before = data_before.squeeze(0).squeeze(0).detach().numpy()
    after = data_after.squeeze(0).squeeze(0).detach().numpy()
    return before, after 


def get_unique_embeddings(model,dataset,idx,skip=False,verbose=True):
    """
    Given model, dataset, and index, this returns the unique embeddings that each voxel 
    in the given input image is being assigned to
    """
    before = dataset[idx][0].unsqueeze(1)
    if skip==False: 
        encoded = model.encoder(before)
    else:
        _,_,_,encoded = model.encoder(before)
    _,_,_,embed = model.quantization(encoded)
    embed = embed.detach().numpy()
    full_shape = embed.shape[1]
    embed_used = np.unique(embed,axis=0)
    if verbose == True:
        print(f"Number of embeddings actually used is {embed_used.shape[0]} out of {full_shape}")
    return(embed_used)


def get_unique_embeddings_SOM(model,dataset,idx,skip=False,verbose=True):
    """
    Same as get_unique_embeddings() but used with the SOMVAE model 
    """
    before = dataset[idx][0].unsqueeze(1)
    if skip==False: 
        encoded = model.encoder(before)
    else:
        _,_,_,encoded = model.encoder(before)
    _,_,embed = model.quantization(encoded)
    embed = embed.detach().numpy()
    full_shape = embed.shape[1]
    embed_used = np.unique(embed,axis=0)
    if verbose == True:
        print(f"Number of embeddings actually used is {embed_used.shape[0]} out of {full_shape}")
    return(embed_used)


def get_TSNE_encodings(model,dataset,idx,skip=False,learning_rate=100):
    """
    Given model and dataset and index, this flattens the encoder output and performs TSNE
    along with the codebook. This helps to get an idea of how close the codebook is to the
    encoder output. 
    """
    before = dataset[idx][0].unsqueeze(1)
    if skip==False: 
        encoded = model.encoder(before)
    else:
        _,_,_,encoded = model.encoder(before)
    flat_encoded = encoded.reshape(encoded .shape[0],encoded.shape[1],-1).permute(0,2,1).reshape(-1,encoded.shape[1]).detach().numpy()
    embedding_wts = model.quantization._embedding.weight.detach().numpy()
    combined = np.vstack((flat_encoded,embedding_wts))
    labels = np.repeat("Dictionary",combined.shape[0])
    labels[0:flat_encoded.shape[0]] = "Image"
    TSNEmodel = TSNE(n_components=2,learning_rate=learning_rate)
    combined_tsne = TSNEmodel.fit_transform(combined)
    combined_tsne_df = pd.DataFrame({"TSNE1":combined_tsne[:,0],
                                    "TSNE2":combined_tsne[:,1],
                                    "Type": labels})
    return(combined_tsne_df)


#a = get_unique_embeddings(chosenmodel, traindataset, 5)
#b = get_TSNE_encodings(vanillamodel, traindataset, 100)

#sns.lmplot("TSNE1","TSNE2",data=b,hue="Type",fit_reg=False)
#plt.show()