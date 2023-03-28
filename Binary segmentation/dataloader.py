from operator import index
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
import torchvision.transforms as T
from torchvision.io import read_image
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import albumentations as A
from PIL import Image

class dataset(Dataset):
    def __init__(self, imagePaths, maskPaths, transform=None):
        self.imagePaths = imagePaths
        self.maskPaths = maskPaths
        self.transform = transform
        
    def __getitem__(self, index):
        x = np.array(Image.open(self.imagePaths[index]).convert("L"), dtype = np.float)
#         x = (x-np.min(x))/(np.max(x)-np.min(x))
        y = cv2.imread(self.maskPaths[index], 0)
        if self.transform:
            aug = self.transform(image = x, mask = y)
            x = aug['image']
            y = aug['mask']
#             mask = torch.max(mask, dim = 2)[0]
#             y = y.type(torch.long)
        return x, y
    
    def __len__(self):
        return len(self.imagePaths)
