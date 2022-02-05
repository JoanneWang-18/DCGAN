import pandas as pd
import numpy as np
import torch
import random

df = pd.read_csv("C:/Users/joann/Desktop/Python/dcgan_celeba/data/whole_combined_2.csv")

tensor = np.zeros((0, 64), float)

def custom_training_set():
    global tensor, df
    for column in range(df.shape[1]):
        column = str(column)
        nan_num = df[column].isna().sum()
        for start in range(df.shape[0]-nan_num):
            if (start + 64) > (df.shape[0]-nan_num):
                break
            #print(df[column][start: start + 60])
            #print(df[column][start: start + 60].shape)
            arr = np.expand_dims(df[column][start: start + 64], axis=1)
            arr = 2. * (arr - np.min(arr)) / np.ptp(arr) - 1
            arr = np.transpose(arr)
            tensor = np.append(tensor, arr, axis=0)
    #print(tensor.shape)
    #print(tensor)

    dataset = torch.from_numpy(tensor)
    dataset = dataset.unsqueeze(1).unsqueeze(-1)
    dataset = dataset.type(torch.FloatTensor)
    dataset = dataset.cuda()
    #print(dataset.shape)
    return dataset

custom_training_set()

#def sampleNoise():
def sampleNoise(bsize, nz):
    global tensor

    noise = np.zeros((0, 20), float)
    for count in range(64):
        index = np.random.randint(0, tensor.shape[0])
        #print(index)
        #print(tensor[index])
        add = np.expand_dims(tensor[index], axis=1)
        add = np.transpose(add)
        noise = np.append(noise, add, axis=0)
    # print("NOISE", noise.shape)
    # print(noise)
    r_tensor = np.asarray(noise)
    r_tensor = r_tensor.astype(float).reshape(64, 20)
    r_tensor = torch.from_numpy(r_tensor)
    r_tensor = r_tensor.unsqueeze(-1).unsqueeze(-1)
    r_tensor = r_tensor.type(torch.FloatTensor)
    r_tensor = r_tensor.cuda()
    #print(r_tensor.shape)
    return r_tensor
