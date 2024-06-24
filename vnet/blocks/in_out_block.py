import torch
import torch.nn as nn
import torch.nn.functional as F

from .convbnact import ConvBnAct3d, ConvBnAct2d


class InputBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_type=nn.BatchNorm3d, act_type=nn.ReLU, num_layers=1):
        super(InputBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            if i == 0:
                conv = ConvBnAct3d(in_channels, out_channels,
                                   kernel_size=5, padding=2, stride=1,
                                   norm_type=norm_type,
                                   act_type=act_type)
            else:
                conv = ConvBnAct3d(in_channels, out_channels,
                                   kernel_size=3, padding=1, stride=1,
                                   norm_type=norm_type,
                                   act_type=act_type)
            layers.append(conv)
        self.conv = nn.Sequential(*layers)

    def forward(self, input):
        out = self.conv(input)
        return out


class OutBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_type=nn.BatchNorm3d, act_type=nn.ReLU):
        super(OutBlock, self).__init__()
        self.conv = nn.Sequential(
            ConvBnAct3d(in_channels, in_channels, norm_type=norm_type, act_type=act_type),
            ConvBnAct3d(in_channels, out_channels, kernel_size=1, padding=0, norm_type=False, act_type=False),
        )

    def forward(self, input):
        out = self.conv(input)
        return out

class OutBlock2d(nn.Module):
    def __init__(self, in_channels, out_channels, norm_type=nn.BatchNorm2d, act_type=nn.ReLU):
        super(OutBlock2d, self).__init__()
        self.conv = nn.Sequential(
            ConvBnAct2d(in_channels, in_channels, norm_type=norm_type, act_type=act_type),
            ConvBnAct2d(in_channels, out_channels, kernel_size=1, padding=0, norm_type=False, act_type=False),
        )

    def forward(self, input):
        out = self.conv(input)
        return out