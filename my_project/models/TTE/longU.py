import torch
from torch import nn
import numpy as np

class Unet2D(nn.Module):
    def __init__(self, in_channels, out_channels, channel_ratio=1):
        super().__init__()
        ch = np.array([32, 64, 128, 256])
        ch = channel_ratio*ch
        ch = ch.round().astype(int)




        self.conv1 = self.contract_block(in_channels, ch[0], 7, 3)
        self.conv2 = self.contract_block(ch[0], ch[1], 3, 1)
        self.conv3 = self.contract_block(ch[1], ch[2], 3, 1)
        self.conv4 = self.contract_block(ch[2], ch[3], 3, 1)

        self.conv5 = self.same_block(ch[3], ch[3], 3, 1)

        self.upconv4 = self.expand_block(ch[3], ch[2], 3, 1)
        self.upconv3 = self.expand_block(2*ch[2], ch[1], 3, 1)
        self.upconv2 = self.expand_block(2*ch[1], ch[0], 3, 1)
        self.upconv1 = self.expand_block(2*ch[0], out_channels, 3, 1)

    def __call__(self, x):
        # downsampling part
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        conv5 = self.conv5(conv4)

        #upsample
        upconv4 = self.upconv4(conv5)
        upconv3 = self.upconv3(torch.cat([upconv4, conv3], 1))
        upconv2 = self.upconv2(torch.cat([upconv3, conv2], 1))
        upconv1 = self.upconv1(torch.cat([upconv2, conv1], 1))

        return upconv1

    def contract_block(self, in_channels, out_channels, kernel_size, padding):

        contract = nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),

            torch.nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),

            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                                 )

        return contract

    def expand_block(self, in_channels, out_channels, kernel_size, padding):

        expand = nn.Sequential(torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
                            torch.nn.BatchNorm2d(out_channels),
                            torch.nn.ReLU(),

                            torch.nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding),
                            torch.nn.BatchNorm2d(out_channels),
                            torch.nn.ReLU(),

                            torch.nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1) 
                            )
        return expand

    def same_block(self, in_channels, out_channels, kernel_size, padding):

        contract = nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),

            torch.nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
                                 )
        return contract
