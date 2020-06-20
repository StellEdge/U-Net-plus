import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.batch_norm(x)
        x = F.relu(self.conv2(x))
        x = self.batch_norm(x)
        return x

class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up4 = nn.Upsample(scale_factor=2, mode='nearest')

        conv_1_out = 32
        conv_2_out = 64
        conv_3_out = 128
        conv_4_out = 256
        conv_5_out = 512

        self.conv1 = ConvBlock(1, conv_1_out)
        self.conv2 = ConvBlock(conv_1_out, conv_2_out)
        self.conv3 = ConvBlock(conv_2_out, conv_3_out)
        self.conv4 = ConvBlock(conv_3_out, conv_4_out)

        self.conv5 = ConvBlock(conv_4_out, conv_5_out)

        self.conv6 = ConvBlock(conv_5_out+conv_4_out, conv_4_out)
        self.conv7 = ConvBlock(conv_4_out+conv_3_out, conv_3_out)
        self.conv8 = ConvBlock(conv_3_out+conv_2_out, conv_2_out)
        self.conv9 = ConvBlock(conv_2_out+conv_1_out, conv_1_out)

        self.conv10 = nn.Conv2d(conv_1_out, 1, 1)

    def forward(self, x):
        c1 = self.conv1(x)
        x = self.pool1(c1)
        c2 = self.conv2(x)
        x = self.pool2(c2)
        c3 = self.conv3(x)
        x = self.pool3(c3)
        c4 = self.conv4(x)
        x = self.pool4(c4)
        x = self.conv5(x)
        x = self.up1(x)
        x = torch.cat([x, c4], 1)
        x = self.conv6(x)
        x = self.up2(x)
        x = torch.cat([x, c3], 1)
        x = self.conv7(x)
        x = self.up3(x)
        x = torch.cat([x, c2], 1)
        x = self.conv8(x)
        x = self.up4(x)
        x = torch.cat([x, c1], 1)
        x = self.conv9(x)
        x = self.conv10(x)
        return x

