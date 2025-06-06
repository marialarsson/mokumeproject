import numpy as np
import torch
import torch.nn as nn

# UNET MODEL base https://qiita.com/gensal/items/03e9a6d0f7081e77ba37

# Conv2d --> BatchNorm2d --> ReLU 2 times
class TwoConvBlock(nn.Module):
    def __init__(self, ch_in, ch_mid, ch_out):
        super().__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_mid, kernel_size = 3, padding="same")
        self.bn1   = nn.BatchNorm2d(ch_mid)
        self.rl    = nn.LeakyReLU() #nn.ReLU() <--the only modification made to the unet (relu--->leaky relu)
        self.conv2 = nn.Conv2d(ch_mid, ch_out, kernel_size = 3, padding="same")
        self.bn2   = nn.BatchNorm2d(ch_out)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.rl(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.rl(x)
        return x

# Upsample --> BatchNorm2d --> Conv2d --> BatchNorm2d
class UpConv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.bn1  = nn.BatchNorm2d(ch_in)
        self.conv = nn.Conv2d(ch_in, ch_out, kernel_size = 3, padding="same")
        self.bn2  = nn.BatchNorm2d(ch_out)

    def forward(self, x):
        x = self.up(x)
        #import ipdb; ipdb.set_trace()
        x = self.bn1(x)
        x = self.conv(x)
        x = self.bn2(x)
        return x

class UNet_2D(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.TCB1 = TwoConvBlock(in_dim, 64,   64) # encoder
        self.TCB2 = TwoConvBlock(  64,  128,  128) # encoder
        self.TCB3 = TwoConvBlock( 128,  256,  256) # encoder
        self.TCB4 = TwoConvBlock( 256,  512,  512) # encoder
        self.TCB5 = TwoConvBlock( 512, 1024, 1024) # Encoder (middle/bottom)
        self.TCB6 = TwoConvBlock(1024,  512,  512) # decoder
        self.TCB7 = TwoConvBlock( 512,  256,  256) # decoder
        self.TCB8 = TwoConvBlock( 256,  128,  128) # decoder
        self.TCB9 = TwoConvBlock( 128,   64,   64) # decoder
        self.maxpool = nn.MaxPool2d(2, stride = 2)

        self.UC1 = UpConv(1024, 512)
        self.UC2 = UpConv( 512, 256)
        self.UC3 = UpConv( 256, 128)
        self.UC4 = UpConv( 128,  64)

        self.conv1 = nn.Conv2d(64, out_dim, kernel_size = 1)

    def forward(self, x):
        x = self.TCB1(x)
        x1 = x
        x = self.maxpool(x)

        x = self.TCB2(x)
        x2 = x
        x = self.maxpool(x)

        x = self.TCB3(x)
        x3 = x
        x = self.maxpool(x)

        x = self.TCB4(x)
        x4 = x
        x = self.maxpool(x)

        x = self.TCB5(x)

        x = self.UC1(x)
        x = self.TCB6(torch.cat([x4, x], dim = 1))
        x = self.UC2(x)
        x = self.TCB7(torch.cat([x3, x], dim = 1))
        x = self.UC3(x)
        x = self.TCB8(torch.cat([x2, x], dim = 1))
        x = self.UC4(x)
        x = self.TCB9(torch.cat([x1, x], dim = 1))
        x = self.conv1(x)

        return x
