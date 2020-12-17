import os
import torch 
import torchvision
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.nn import Conv2d , MaxPool2d , ConvTranspose3d ,Upsample

class LinearAutoencoder(nn.Module):
    def __init__(self):
        super(LinearAutoencoder, self).__init__()
        ## encoder ##
        self.enc1 = nn.Linear(in_features=784, out_features=256)
        self.enc2 = nn.Linear(in_features=256, out_features=128)
        self.enc3 = nn.Linear(in_features=128, out_features=64)
        self.enc4 = nn.Linear(in_features=64, out_features=32)
        self.enc5 = nn.Linear(in_features=32, out_features=16)
        # decoder 
        self.dec1 = nn.Linear(in_features=16, out_features=32)
        self.dec2 = nn.Linear(in_features=32, out_features=64)
        self.dec3 = nn.Linear(in_features=64, out_features=128)
        self.dec4 = nn.Linear(in_features=128, out_features=256)
        self.dec5 = nn.Linear(in_features=256, out_features=784)

    def forward(self, x):
        # Encoder
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.enc3(x))
        x = F.relu(self.enc4(x))
        x = F.relu(self.enc5(x))
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = F.relu(self.dec3(x))
        x = F.relu(self.dec4(x))
        x = F.relu(self.dec5(x))
        return x

# initialize the NN
# encoding_dim = 32
class CNN_Autoencoder(nn.Module):
    def __init__(self):
        super(CNN_Autoencoder, self).__init__()
        self.encoder = nn.Sequential(Conv2d(1,16,kernel_size=3,stride=1),
                                     nn.ReLU(inplace=True),
                                     MaxPool2d(2,2),

                                     Conv2d(16,8,kernel_size=3,stride=1),
                                     nn.ReLU(inplace=True),
                                     MaxPool2d(2,2),
                                     Conv2d(8,8,kernel_size=3,stride=1))

        self.decoder = nn.Sequential(

            ConvTranspose3d(8, 16, kernel_size=3, stride=1),
            nn.Upsample(scale_factor=(2, 3.5, 2)),
            nn.ReLU(inplace=True),


            ConvTranspose3d(16, 32, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            ConvTranspose3d(32, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),

            ConvTranspose3d(64,1 , kernel_size=3, stride=1),
            nn.ReLU(inplace=True),






        )

    def forward(self,x):
        x = self.encoder(x)
        x = x.view(x.shape[0],-1).view(x.shape[0],8,3,1,3)
        #print(x.shape)
        x = self.decoder(x)
        #print(x.shape)
        return x



if __name__ == '__main__':
    CNN = CNN_Autoencoder()
    print(CNN)