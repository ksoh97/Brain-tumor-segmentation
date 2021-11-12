import matplotlib.pyplot as plt
import numpy as np
import random
import SimpleITK as sitk  # For loading the dataset
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
import math

def read_img(img_path):
    """
    Reads a .nii.gz image and returns as a numpy array.
    """    
    return sitk.GetArrayFromImage(sitk.ReadImage(img_path))

def get_datapath(datadir, test_size = 0.25):
    dirs = []
    images = []
    masks = []
    for dirname, _, filenames in os.walk(datadir):
        for filename in filenames:
            if 'mask'in filename:
                dirs.append(dirname.replace(datadir, ''))
                masks.append(filename)
                images.append(filename.replace('_mask', ''))

    image_list = []
    mask_list = []
    for i in range(len(dirs)):  
        imagePath = os.path.join(datadir, dirs[i], images[i])
        maskPath = os.path.join(datadir, dirs[i], masks[i])

        # before
        # image_list.append(datadir+imagePath)
        # mask_list.append(datadir+maskPath)

        # after
        image_list.append(imagePath)
        mask_list.append(maskPath)
    return image_list, mask_list



class DataSegmentationLoader(Dataset):
    def __init__(self, path_list,ground_list = []):
        self.sample = path_list
        self.ground_truth = []
        if len(ground_list) > 0:
            self.ground_truth = ground_list
            
        self.image_transform = transforms.Compose([                                
                                transforms.Resize((256, 256)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                            ])
        self.label_transform = transforms.Compose([                                
                                transforms.Resize((256, 256)),
                                transforms.ToTensor()                                
                            ])        

    def __len__(self):
        return len(self.sample)

    def __getitem__(self, idx):
        #Load Data        
        data = Image.open(self.sample[idx])
        data = self.image_transform(data)                
        if len(self.ground_truth) > 0:
            label = Image.open(self.ground_truth[idx])
            label = self.label_transform(label)        
        else:
            label = np.zeros((1,256,256))
        
        return data, label