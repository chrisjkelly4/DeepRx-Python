import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim




class ResNetBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

        padding_val = (dilation[0], dilation [1])
        self.relu = nn.ReLU() 
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.depthConv1 = nn.Conv2d(in_channels, in_channels * 2, kernel_size=3, groups=in_channels, padding=padding_val, dilation=dilation)
        self.shapeConv1 = nn.Conv2d(in_channels * 2, out_channels, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.depthConv2 = nn.Conv2d(out_channels, out_channels * 2, kernel_size=3, groups=out_channels, padding=padding_val, dilation = dilation )
        self.shapeConv2 = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1)
        
        
    def forward(self, input):
        shortcut = self.shortcut(input)
        normalised_input = self.bn1(input)
        a1 = self.relu(normalised_input)
        depth1 = self.depthConv1(a1)
        shape2 = self.shapeConv1(depth1)
        normalised_input2 = self.bn2(shape2)
        a2 = self.relu(normalised_input2)
        depth2 = self.depthConv2(a2)
        shape2 = self.shapeConv2(depth2)
        
        return shape2 + shortcut
    
class DeepRx(nn.Module):
    def __init__(self, number_receivers):
        super().__init__()

        # CHANNEL_SIZES = [64, 64, 64, 128, 128, 256, 256, 256, 128, 128, 64, 64]

        in_channels = 2 * (2 * number_receivers + 1)

        self.ConvIn = nn.Conv2d(in_channels,64,kernel_size=3,padding=1, dilation=(1,1))
        self.resNet_blocks = nn.ModuleList([
            ResNetBlock(64, 64, (1,1)),
            ResNetBlock(64, 128, (1, 1)),
            ResNetBlock(128, 128, (2, 3)),
            ResNetBlock(128, 256, (2, 3)),
            ResNetBlock(256, 256, (2, 3)),
            ResNetBlock(256, 256, (3, 6)),
            ResNetBlock(256, 256, (2, 3)),
            ResNetBlock(256, 128, (2, 3)),
            ResNetBlock(128, 128, (2, 3)),
            ResNetBlock(128, 64, (1, 1)),
            ResNetBlock(64, 64, (1, 1)),

        ])
        # channel_size_last_index = CHANNEL_SIZES[-1]
        self.ConvOut = nn.Conv2d(64 ,4,kernel_size=3,padding=1, dilation=(1,1))

    def forward(self, input):
        x = self.ConvIn(input)
        for block in self.resNet_blocks:
            x = block(x)
        x = self.ConvOut(x)

        return x


