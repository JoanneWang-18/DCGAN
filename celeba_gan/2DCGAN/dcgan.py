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

        """self.conv1 = nn.Conv2d(params['nz'], 1, kernel_size=(1, 1), stride=(1, 3), padding=(1, 3))
        self.bn = nn.BatchNorm2d(1, 0.8)"""

        self.conv1 = nn.ConvTranspose2d(params['nz'], params['ngf']*8, kernel_size=(4, 1), stride=(2, 2), padding=0)
        self.bn = nn.BatchNorm2d(params['ngf']*8, 0.8)

        self.conv2 = nn.ConvTranspose2d(params['ngf']*8, params['ngf']*4, kernel_size=(5, 1), stride=(1, 2), padding=0)
        self.bn1 = nn.BatchNorm2d(params['ngf']*4, 0.8)

        self.conv3 = nn.ConvTranspose2d(params['ngf']*4, params['ngf']*2, kernel_size=(2, 1), stride=(2, 2), padding=0)
        self.bn2 = nn.BatchNorm2d(params['ngf']*2, 0.8)

        self.conv4 = nn.ConvTranspose2d(params['ngf']*2, params['ngf'], kernel_size=(2, 1), stride=(2, 2), padding=0)
        self.bn3 = nn.BatchNorm2d(params['ngf'], 0.8)

        self.conv5 = nn.ConvTranspose2d(params['ngf'], params['nc'], kernel_size=(2, 1), stride=(2, 2), padding=0)

        """self.fc1 = nn.Linear(in_features=9, out_features=30)
        self.bn3 = nn.BatchNorm1d(30, 0.8)
        self.drop1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(in_features=30, out_features=40)
        self.bn4 = nn.BatchNorm1d(40, 0.8)
        self.drop2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(in_features=40, out_features=50)
        self.bn5 = nn.BatchNorm1d(50, 0.8)
        self.drop3 = nn.Dropout(0.3)

        self.fc4 = nn.Linear(in_features=50, out_features=60)"""

    def forward(self, x):
        #x = x.view(x.size(0), -1)
        print("G", x.shape)
        x = F.relu(self.bn(self.conv1(x)))
        print("G", x.shape)
        x = F.relu(self.bn1(self.conv2(x)))
        print("G", x.shape)
        x = F.relu(self.bn2(self.conv3(x)))
        print("G", x.shape)
        x = F.relu(self.bn3(self.conv4(x)))
        print("G", x.shape)
        x = torch.tanh(self.conv5(x))
        print("G", x.shape)

        #x = torch.flatten(x, 1)
        #print("G flatten", x.shape)
        """x = F.relu(self.bn3(self.fc1(x)))
        x = F.dropout(self.drop1(x))
        #print("G", x.shape)
        x = F.relu(self.bn4(self.fc2(x)))
        x = F.dropout(self.drop2(x))
        #print("G", x.shape)
        x = F.relu(self.bn5(self.fc3(x)))
        x = F.dropout(self.drop3(x))
        #print("G", x.shape)
        x = torch.tanh(self.fc4(x))
        #print("G", x.shape)"""
        return x


class Discriminator(nn.Module):
    def __init__(self, params):
        super().__init__()

        self.conv1 = nn.Conv2d(params['nc'], params['ndf'], kernel_size=(2, 1), stride=(2, 2), padding=0)
        self.bn = nn.BatchNorm2d(params['ndf'], 0.8)

        self.conv2 = nn.Conv2d(params['ndf'], params['ndf']*2, kernel_size=(2, 1), stride=(2, 2), padding=0)
        self.bn1 = nn.BatchNorm2d(params['ndf']*2, 0.8)

        self.conv3 = nn.Conv2d(params['ndf']*2, params['ndf']*4, kernel_size=(2, 1), stride=(2, 2), padding=0)
        self.bn2 = nn.BatchNorm2d(params['ndf']*4, 0.8)

        self.conv4 = nn.Conv2d(params['ndf']*4, 1, kernel_size=(8, 1), stride=(2, 2), padding=0)
        #self.bn3 = nn.BatchNorm2d(params['ndf']*8, 0.8)

        #self.conv5 = nn.Conv2d(params['ndf']*8, 1, kernel_size=(4, 1), stride=(2, 2), padding=0)

        """self.fc1 = nn.Linear(in_features=60, out_features=50)
        self.bn1 = nn.BatchNorm1d(50, 0.8)

        self.fc2 = nn.Linear(in_features=50, out_features=40)
        self.bn2 = nn.BatchNorm1d(40, 0.8)

        self.fc3 = nn.Linear(in_features=40, out_features=30)
        self.bn3 = nn.BatchNorm1d(30, 0.8)

        self.fc4 = nn.Linear(in_features=30, out_features=1)"""

    def forward(self, x):
        print("D:", x.shape)
        x = F.leaky_relu(self.bn(self.conv1(x)), 0.2, True)
        print("D:", x.shape)
        x = F.leaky_relu(self.bn1(self.conv2(x)), 0.2, True)
        print("D:", x.shape)
        x = F.leaky_relu(self.bn2(self.conv3(x)), 0.2, True)
        print("D:", x.shape)
        x = torch.sigmoid(self.conv4(x))
        print("D:", x.shape)
        """x = F.leaky_relu(self.bn3(self.conv4(x)), 0.2, True)
        print("D:", x.shape)
        x = torch.sigmoid(self.conv5(x))
        print("D:", x.shape)"""

        """x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.bn1(self.fc1(x)), 0.2, True)
        print("D:", x.shape)
        x = F.leaky_relu(self.bn2(self.fc2(x)), 0.2, True)
        print("D:", x.shape)
        x = F.leaky_relu(self.bn3(self.fc3(x)), 0.2, True)
        print("D:", x.shape)
        x = torch.sigmoid(self.fc4(x))
        print("D:", x.shape)"""
        return x
