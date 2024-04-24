from .multikernel_conv_block import MultiKernelConvBlock
import torch
import torch.nn.functional as F
from einops import rearrange
import math


class DNAEncoder(torch.nn.Module):
    """DNAEncoder use several consecutive ConvMaxpool to encode DNA sequence
    input: a 4xN matrix, where N is the length of DNA sequence
    output: a KxM matrix, where K is the increased Channel number and M is
            the reduced length of DNA sequence M = N/2^K
    details: a wrapper of ConvBlock; check ConvBlock for more details
    """

    def __init__(self, num_layers=3, target_width=64):
        super().__init__()
        # increse the width of DNA sequence from 4 -> target_width//2
        self.feature_layer = ConvBlock([4, target_width // 2], [15])
        # increase the width of DNA sequence from
        # target_width//2 -> target_width
        # and use maxpool to reduce the length of DNA sequence by half
        tower_filter_list = exponential_linspace_int(
            target_width // 2,
            target_width,
            num=(num_layers + 1),
            divisible_by=2,
        )
        self.layers = ConvBlock(tower_filter_list, [5], maxpool=True)

    def forward(self, x):
        x = self.feature_layer(x)
        x = self.layers(x)
        return x


class ConvBlock(torch.nn.Module):
    """ConvBlock use maxPool+CNN to encode DNA sequence to a two
    dimension vector:
        1. reduce the length dimension of DNA sequence by half
        2. use CNN to encode DNA sequence to increase
           the dimension of DNA sequence
    Other types of encoder can be added by inheriting this class
    Args:
        filter_list: a list of intergers, specifying the number of input
                     and output channels
        kernel_size: a list of integers, specifying the kernel size, default
                     to [5]
    return:
        a 2D vector of DNA sequence
    """

    def __init__(self, filter_list, kernel_size, maxpool=False):
        super().__init__()

        conv_layers = []
        for dim_in, dim_out in zip(filter_list[:-1], filter_list[1:]):
            conv_layers.append(
                torch.nn.Sequential(
                    MultiKernelConvBlock(
                        dim_in, dim_out, kernel_sizes=kernel_size
                    ),
                    Residual(MultiKernelConvBlock(dim_out, dim_out, [1])),
                )
            )
            if maxpool:
                conv_layers.append(MaxPool())
        self.conv_tower = torch.nn.Sequential(*conv_layers)

    def forward(self, x):
        return self.conv_tower(x)


class Residual(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


def exponential_linspace_int(start, end, num, divisible_by=1):
    def _round(x):
        return int(round(x / divisible_by) * divisible_by)

    base = math.exp(math.log(end / start) / (num - 1))
    return [_round(start * base**i) for i in range(num)]


class MaxPool(torch.nn.Module):
    def __init__(self, kernel_size=2, stride=2):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        x = rearrange(x, "b l d-> b d l")
        x = F.max_pool1d(x, self.kernel_size, self.stride)
        x = rearrange(x, "b d l -> b l d")
        return x
