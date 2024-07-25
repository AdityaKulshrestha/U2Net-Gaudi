from tqdm import tqdm 
import torch 
import torch.nn as nn 
from torch.nn import functional as F 
import habana_frameworks.torch.core as htcore





class Encoder(nn.Module):
    """Encoder module of Unet"""
    def __init__(self, in_channels, out_channels, num_filters):
        super.__init__()
        self.conv1 = nn.Conv2d()
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d() 
        self.act2 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2D() 

    def forward(self, x): 
        x = self.conv1(x) 
        x = self.act1(x) 
        x = self.conv(x) 
        x = self.act2(x) 
        x = self.maxpool1(x) 
        return x 



class Decoder(nn.Module): 
    """Decoder module of the Unet"""
    def __init__(self, ):
        super().__init__()
        self.upsample1 = nn.Upsample() 
        self.conv3 = nn.Conv2d() 
        self.act3 = nn.ReLU() 
        self.conv4 = nn.Conv2d() 
        self.act4 = nn.ReLU() 

    def forward(self, x):
        

