import os
import numpy as np 
import pandas as pd 
import nibabel as nib


import torch 
from torch.utils.data import Dataset, DataLoader
import torchio as tio 

#This module contains the dataset classes to load in the OASIS data
#Locally the data will be streamed in but for colab this is very slow 
#For colab, all data is brought into RAM

class OASISDataset_Local(Dataset):
    def __init__(self,foldername,annotations_data):
        self.folder = foldername
        self.filedata = annotations_data
        self.fileinfo = np.array(self.filedata.filelist)
        self.standardizer = tio.ZNormalization()
        self.padder = tio.CropOrPad((144,176,144))
        #self.fileinfo = pd.read_csv(annotations_file)
    
    def __len__(self):
        return len(self.fileinfo)
    
    def __getitem__(self,idx):
        img_path = os.path.join(self.folder,self.fileinfo[idx])
        img = torch.from_numpy(nib.load(img_path).get_fdata()).float().unsqueeze(0)
        img = self.standardizer(img)
        img = self.padder(img)
        #img = nib.load(img_path)#.get_fdata()
        return img, img 

class OASISDataset_Colab(Dataset):
    def __init__(self,foldername,annotations_data):
        self.folder = foldername
        self.filedata = annotations_data
        self.fileinfo = np.array(self.filedata.filelist)
        self.standardizer = tio.ZNormalization()
        self.padder = tio.CropOrPad((144,176,144))
        
        self.images = []
        for i in range(len(self.fileinfo)):
          self.images.append(torch.from_numpy(nib.load(os.path.join(self.folder,self.fileinfo[i])).get_fdata()).float().unsqueeze(0))
        
    
    def __len__(self):
        return len(self.fileinfo)
    
    def __getitem__(self,idx):
        
        img = self.images[idx]
        img = self.standardizer(img)
        img = self.padder(img)
        
        return img, img 