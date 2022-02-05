from __future__ import print_function, division
import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms
import noise_gen
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler


'''class CelebaDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform):
        df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.csv_path = csv_path
        self.img_names = df.index.values
        self.y = df['Male'].values
        self.transform = transform

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, index):
        index = str(index+1)
        index = index.zfill(6)
        #print(os.path.join(self.img_dir, index + ".jpg"))
        img = Image.open(os.path.join(self.img_dir, index + ".jpg"))

        if self.transform is not None:
            img = self.transform(img)

        index = int(index)
        label = self.y[index]
        return img, label'''

def get_celeba(params):
#def get_celeba():
    """
    Loads the dataset and applies proproccesing steps to it.
    Returns a PyTorch DataLoader.
    """
    '''
    # Data proprecessing.
    transform = transforms.Compose([
        transforms.Resize(params['imsize']),
        transforms.CenterCrop(params['imsize']),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))])
    '''

    # Create the dataset.
    #dataset = datasets.ImageFolder(root=root, transform=transform)
    #dataset = CelebaDataset(csv_path=r"C:\Users\joann\Desktop\Python\dcgan_celeba\data\list_attr_celeba.csv",
                            #img_dir=r"C:\Users\joann\Desktop\Python\dcgan_celeba\data\img_align_celeba",
                            #transform=transform)
    my_dataset = noise_gen.custom_training_set()

    dataset = TensorDataset(my_dataset)

    #for img, label in dataset:
        #print("Image shape:", img.shape, ", Label:", label)
    # Create the dataloader.
    dataloader = torch.utils.data.DataLoader(dataset,
                                             #batch_size=1,
                                             batch_size=params['bsize'],
                                             shuffle=False,
                                             drop_last=True)

    #for i, batch in enumerate(dataloader):
        #print(i, batch[0])
        # has labels: batch[1]

    return dataloader
