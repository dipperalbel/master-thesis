import torch

'''ShuffleNetV2 in PyTorch.

See the paper "ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design" for more details.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import init
import math


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv3d(inp, oup, kernel_size=3, stride=stride, padding=(1,1,1), bias=False),
        nn.BatchNorm3d(oup),
        nn.ReLU(inplace=True)
    )

def conv_1x1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv3d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm3d(oup),
        nn.ReLU(inplace=True)
    )

def conv_3x3x3_dw(inp, oup):
    return nn.Sequential(
        nn.Conv3d(
            in_channels = inp, 
            out_channels=oup, 
            kernel_size=3, 
            groups=inp,
            stride=2, 
            padding=1),
        nn.BatchNorm3d(oup),
        nn.ReLU(inplace=True)
    )



def channel_shuffle(x, groups):
    '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
    batchsize, num_channels, depth, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups, 
        channels_per_group, depth, height, width)
    #permute
    x = x.permute(0,2,1,3,4,5).contiguous()
    # flatten
    x = x.view(batchsize, num_channels, depth, height, width)
    return x

class Shuffle(nn.Module):
    def __init__(self, split_size, in_channels, out_channels):
        super(Shuffle, self).__init__()
        in_chan_1 = split_size
        in_chan_2 = in_channels - split_size

        out_chan_1 = int(split_size / in_channels * out_channels)
        out_chan_2 = out_channels - out_chan_1


        self.branch1 = nn.Sequential(
            conv_3x3x3_dw(in_chan_1, in_chan_1), 
            conv_1x1x1_bn(in_chan_1, out_chan_1))

        self.branch2 = nn.Sequential(
            conv_1x1x1_bn(in_chan_2, in_chan_2), 
            conv_3x3x3_dw(in_chan_2, in_chan_2), 
            conv_1x1x1_bn(in_chan_2, out_chan_2))
        
        self.c = split_size

    def forward(self, x):
        b1_x = x[:, :self.c]
        b2_x = x[:, self.c:]
        b1_output = self.branch1(b1_x)
        b2_output = self.branch2(b2_x)
        out = torch.cat([b1_output, b2_output], dim=1)
        output = channel_shuffle(out, 4)
        return output

if __name__ == '__main__':
    sh = Shuffle(10, 100, 100)
    print(sh)