import pandas as pd
import numpy as np
import torch
import random
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
parser.add_argument('-load_path', default=r'C:\Users\joann\Desktop\Python\dcgan_celeba\model_final.pth',
                    help='Checkpoint to load path from')
# parser.add_argument('-load_path', default=r'C:\Users\joann\Desktop\Python\model_epoch_38.pth', help='Checkpoint to load path from')
parser.add_argument('-num_output', default=64, help='Number of generated outputs')
args = parser.parse_args()

# Load the checkpoint file.
state_dict = torch.load(args.load_path)

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
params = state_dict['params']

# Create the generator network.
netG = Generator(params).to(device)
# Load the trained generator weights.
netG.load_state_dict(state_dict['generator'])

prevbatch = 64
prevnz = 100

generated_img = np.zeros([])
def generate():
    global generated_img
    noise = torch.randn(prevbatch, prevnz, 1, 1, device=device)
    # Turn off gradient calculation to speed up the process.
    with torch.no_grad():
        generated_img = netG(noise).detach().cpu()
        generated_img = np.asarray(generated_img)
        #print(generated_img)
        #print(generated_img.shape)
    return generated_img

generate()

#def sampleNoise_2nd():
def sampleNoise_2nd(params):
    global generated_img

    """noise = np.zeros((0, 60), float)
    for count in range(generated_img.shape[0]):
        single = np.expand_dims(generated_img[count], axis=1)
        single = np.transpose(single)
        noise = np.append(noise, single, axis=0)
    #print("NOISE", noise)
    #print(noise.shape)"""

    r_tensor = np.zeros((0, 64), float)
    for count in range(params['bsize']//generated_img.shape[0]):
        add = generate()
        add = np.squeeze(add, axis=(1, 3))
        #print(add)
        r_tensor = np.append(r_tensor, add, axis=0)
    #print(r_tensor.shape)

    r_tensor = torch.from_numpy(r_tensor)
    r_tensor = r_tensor.unsqueeze(-1).unsqueeze(-1)
    r_tensor = r_tensor.type(torch.FloatTensor)
    r_tensor = r_tensor.cuda()
    #print(r_tensor.shape)
    return r_tensor
