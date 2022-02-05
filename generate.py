import argparse
import pandas as pd
import torch
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import random
import noise_gen
import pandas as pd

from dcgan import Generator

parser = argparse.ArgumentParser()
parser.add_argument('-load_path', default=r'C:\Users\joann\Desktop\Python\dcgan_celeba\model_epoch_4.pth', help='Checkpoint to load path from')
#parser.add_argument('-load_path', default=r'C:\Users\joann\Desktop\Python\model_epoch_38.pth', help='Checkpoint to load path from')
parser.add_argument('-num_output', default=64, help='Number of generated outputs')
args = parser.parse_args()

# Load the checkpoint file.
state_dict = torch.load(args.load_path)

device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
params = state_dict['params']

# Create the generator network.
netG = Generator(params).to(device)
# Load the trained generator weights.
netG.load_state_dict(state_dict['generator'])
print(netG)

#noise = torch.randn(int(args.num_output), params['nz'], 1, 1, device=device)

def realtest():
    tensor = noise_gen.sampleNoise(params['bsize'], params['nz'])
    """df = pd.read_csv("C:/Users/joann/Desktop/Python/dcgan_celeba/data/whole_combined.csv", usecols=['attention'])
    df = 2. * (df - np.min(df)) / np.ptp(df) - 1
    df = df.to_numpy()

    dfArr = np.asarray(df)

    dfArr = dfArr.tolist()
    tensor = random.sample(dfArr, 200)
    tensor = np.asarray(tensor)
    tensor = tensor.astype(float).reshape(4, 50)

    tensor = torch.from_numpy(tensor)
    tensor = tensor.unsqueeze(-1).unsqueeze(-1)
    tensor = tensor.type(torch.FloatTensor)
    tensor = tensor.cuda()"""
    return tensor

def to2ndDCGAN():
    return generated_img

def tocsv(generated_img):
    for batch in range(generated_img.shape[0]):
        for row in range((generated_img.shape[2]-1)):
            print(generated_img[batch][0][row])
            #np.savetxt("C:/Users/joann/Desktop/Python/dcgan_celeba/data/generated.csv", generated_img[batch][0][row], fmt='%1.9f')

noise = realtest()
# Turn off gradient calculation to speed up the process.
with torch.no_grad():
    generated_img = netG(noise).detach().cpu()
    generated_img = np.asarray(generated_img)
    tocsv(generated_img)

"""print(generated_img)
print(generated_img[0].shape)
print(generated_img[0][0].shape)
print(generated_img[0][0][0].shape)
print("0", generated_img[0][0][0])"""


    
