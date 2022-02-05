import torch
import torch.nn as nn
import torch.nn.functional as F


def weights_init(w):
    """
    Initializes the weights of the layer, w.
    """
    classname = w.__class__.__name__
    if classname.find('conv') != -1:
        nn.init.normal_(w.weight.data, 0.0, 0.02)
    elif classname.find('bn') != -1:
        nn.init.normal_(w.weight.data, 1.0, 0.02)
        nn.init.constant_(w.bias.data, 0)


class Generator(nn.Module):
    def __init__(self, params):
        super().__init__()

        # Input is the latent vector Z.
        self.tconv1 = nn.ConvTranspose2d(params['nz'], params['ngf']*8,
            kernel_size=4, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(params['ngf']*8)
        self.drop1 = nn.Dropout(0.5)

        self.tconv2 = nn.ConvTranspose2d(params['ngf']*8, params['ngf']*4,
            4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(params['ngf']*4)
        self.drop2 = nn.Dropout(0.5)

        self.tconv3 = nn.ConvTranspose2d(params['ngf']*4, params['ngf']*2,
            4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(params['ngf']*2)
        self.drop2 = nn.Dropout(0.5)

        self.tconv4 = nn.ConvTranspose2d(params['ngf']*2, params['ngf'],
            4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(params['ngf'])
        self.drop2 = nn.Dropout(0.5)

        self.tconv5 = nn.ConvTranspose2d(params['ngf'], params['nc'],
                                         4, 2, 1, bias=False)
        # state size: (nc) x 128 x 128


    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.tconv1(x)), 0.2, True)
        #print("G", x.shape)
        x = F.leaky_relu(self.bn2(self.tconv2(x)), 0.2, True)
        #print("G", x.shape)
        x = F.leaky_relu(self.bn3(self.tconv3(x)), 0.2, True)
        #print("G", x.shape)
        x = F.leaky_relu(self.bn4(self.tconv4(x)), 0.2, True)
        #print("G", x.shape)
        x = torch.tanh(self.tconv5(x))
        #print("G", x.shape)

        return x

# Define the Discriminator Network
class Discriminator(nn.Module):
    def __init__(self, params):
        super().__init__()

        self.conv1 = nn.Conv2d(params['nc'], params['ndf'],
            4, 2, 1, bias=False)

        self.conv2 = nn.Conv2d(params['ndf'], params['ndf']*2,
            4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(params['ndf']*2)
        self.drop1 = nn.Dropout(0.5)

        self.conv3 = nn.Conv2d(params['ndf']*2, params['ndf']*4,
            4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(params['ndf']*4)
        self.drop2 = nn.Dropout(0.5)

        self.conv4 = nn.Conv2d(params['ndf']*4, params['ndf']*8,
            4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(params['ndf']*8)
        self.drop3 = nn.Dropout(0.5)

        # Input Dimensions: (ndf*8) x 4 x 4
        self.conv5 = nn.Conv2d(params['ndf'] * 8, 1, 4, 1, 0, bias=False)


    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2, True)
        #print("D", x.shape)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2, True)
        #print("D", x.shape)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2, True)
        #print("D", x.shape)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2, True)
        #print("D", x.shape)
        x = torch.sigmoid(self.conv5(x))
        #print("D", x.shape)

        return x