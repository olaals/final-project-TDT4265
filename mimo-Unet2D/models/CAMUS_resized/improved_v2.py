import torch
from torch import nn

class Unet2D(nn.Module):
    def __init__(self, in_channels, out_channels, channel_ratio=1):
        super().__init__()

        self.conv1 = self.contract_block(in_channels, 32*channel_ratio, 7, 3)
        self.conv2 = self.contract_block(32*channel_ratio, 64*channel_ratio, 3, 1)
        self.conv3 = self.contract_block(64*channel_ratio, 128*channel_ratio, 3, 1)
        self.conv4 = self.contract_block(128*channel_ratio, 256*channel_ratio, 3, 1)

        self.upconv4 = self.expand_block(256*channel_ratio, 128*channel_ratio, 3, 1)
        self.upconv3 = self.expand_block(128*2*channel_ratio, 64*channel_ratio, 3, 1)
        self.upconv2 = self.expand_block(64*2*channel_ratio, 32*channel_ratio, 3, 1)
        self.upconv1 = self.expand_block(32*2*channel_ratio, out_channels, 3, 1)

    def __call__(self, x):
        # downsampling part
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        #upsample
        upconv4 = self.upconv4(conv4)
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
