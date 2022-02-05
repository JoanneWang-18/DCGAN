from __future__ import print_function, division
from torch.utils.data import Dataset, DataLoader
import io
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets
import pandas as pd

def get_celeba(params, dset=None):
    """
    Loads the dataset and applies proproccesing steps to it.
    Returns a PyTorch DataLoader.
    """
    # Data proprecessing.
    transform = transforms.Compose([
        transforms.Resize(params['imsize']),
        transforms.CenterCrop(params['imsize']),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5))])

    # Create the dataset.
    root = r"C:\Users\joann\Desktop\Python\dcgan_celeba\data\img_align_celeba"
    dataset = datasets.ImageFolder(root=root, transform=transform)
    #dataset = CustomDataset(csv_file=r"C:\Users\Joanne Wang\Desktop\Python\dcgan\data\list_attr_celeba.csv",
                            #root_dir=r"C:\Users\Joanne Wang\Desktop\Python\dcgan\data\celeba\img_align_celeba")


    # Create the dataloader.
    #list = [for idx in range(len(labels)) if labels[idx]=="1"]
    dataloader = torch.utils.data.DataLoader(dataset,
                                            batch_size=params['bsize'],
                                            shuffle=False)

    return dataloader


