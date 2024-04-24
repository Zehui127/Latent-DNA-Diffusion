import torch
import torch.nn.functional as F
from einops import rearrange
import math
try:
    from .multikernel_conv_block import MultiKernelConvBlock
except:
    from multikernel_conv_block import MultiKernelConvBlock

class SequenceAE(torch.nn.Module):
    """
    An auto-encoder for DNA sequences.
    Encoder upsamples the width dimension, and downsamples in the length dimension.
    Decoder reverses the convolution and pooling process.
    """
    def __init__(self, num_layers=3, funnel_width=64):
        super().__init__()
        self.encoder = DNAEncoder(num_layers=num_layers, target_width=funnel_width)
        self.decoder = DNADecoder(num_layers=num_layers, target_width=funnel_width)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        encoded = self.encode(x)
        y = self.decode(encoded)
        # print("FORWARD", y.size(), x.size())
        return y

    def loss_function(self, recon_x, x):
        MSE = torch.nn.MSELoss(reduction="mean")
        recon_loss = MSE(recon_x, x)
        return recon_loss

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
        self.layers = ConvBlock(tower_filter_list, [5], maxpool=True, upsample=False)

    def forward(self, x):
        x = self.feature_layer(x)
        x = self.layers(x)
        return x

class DNADecoder(torch.nn.Module):
    """DNADecoder decode 2D vector into 1D sequence
    input: a KxM matrix
    output: a 4xN matrix, where N is the length of DNA sequence
    details: a wrapper of ConvBlock; check ConvBlock for more details
    """

    def __init__(self, num_layers=3, target_width=64):
        super().__init__()
        # decrease the width of DNA sequence from
        # target_width -> target_width // 2
        # and use (maybe pseudo) maxpool to increase the length of DNA sequence by half
        tower_filter_list = exponential_linspace_int(
            target_width // 2,
            target_width,
            num=(num_layers + 1),
            divisible_by=2,
        )
        self.layers = ConvBlock(tower_filter_list[::-1], [5], maxpool=False, upsample=True)

        # decrease the width of DNA sequence from target_width//2 -> 4
        self.feature_layer = ConvBlock([target_width // 2, 4], [15])
        # self.last_layer = torch.nn.Conv1d(4, 4, kernel_size=3,
        #                                   stride=1,
        #                                   padding=1)

    def forward(self, x):
        x = self.layers(x)
        x = self.feature_layer(x)
        # x = rearrange(x, "b s d -> b d s")
        # x = self.last_layer(x)
        # x = rearrange(x, "b d s -> b s d")
        # print("Decoder output shape", x.size())
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

    def __init__(self, filter_list, kernel_size, maxpool=False, upsample=False):
        super().__init__()
        self.maxpool = maxpool
        self.upsample = upsample
        self.maxpool_layer = MaxPool()
        self.upsample_layer = UpSample()

        conv_layers = []
        for dim_in, dim_out in zip(filter_list[:-1], filter_list[1:]):
            conv_layers.append(
                torch.nn.Sequential(
                    MultiKernelConvBlock(
                        dim_in, dim_out, kernel_sizes=kernel_size, transpose=upsample # Conv Tranpose when upsampling
                    ),
                    Residual(MultiKernelConvBlock(dim_out, dim_out, [1])),
                )
            )
        self.conv_tower = torch.nn.ModuleList(conv_layers)

    def addPooling(self, x, maxpool=False, upsample=False):
        if maxpool == True:
            for layer in self.conv_tower:
                x = layer(x)
                x = self.maxpool_layer(x)
            return x
        elif upsample == True:
            for layer in self.conv_tower:
                x = self.upsample_layer(x)
                x = layer(x)
            return x
        else:
            for layer in self.conv_tower:
                x = layer(x)
            return x

    def forward(self, x):
        return self.addPooling(x, maxpool=self.maxpool, upsample=self.upsample)


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

# fixed size & stride that halves the length each time
class MaxPool(torch.nn.Module):
    def __init__(self, kernel_size=2, stride=2):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        if x.dim() == 4:
            x = x.squeeze(1)
        x = rearrange(x, "b l d-> b d l")
        x = F.max_pool1d(x, self.kernel_size, self.stride, return_indices=False)
        x = rearrange(x, "b d l -> b l d")
        return x


# fixed size & stride that halves the length each time
class AvgPool(torch.nn.Module):
    def __init__(self, kernel_size=2, stride=2):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        if x.dim() == 4:
            x = x.squeeze(1)
        x = rearrange(x, "b l d-> b d l")
        x = F.avg_pool1d(x, self.kernel_size, self.stride, return_indices=False)
        x = rearrange(x, "b d l -> b l d")
        return x

# class MaxPool(torch.nn.Module):
#     def __init__(self, kernel_size=2, stride=2):
#         super().__init__()
#         self.kernel_size = kernel_size
#         self.stride = stride

#     def forward(self, x):
#         if x.dim() == 4:
#           x = x.squeeze(1)
#         x = rearrange(x, "b l d-> b d l")
#         x, indices = F.max_pool1d(x, self.kernel_size, self.stride, return_indices=True)
#         x = rearrange(x, "b d l -> b l d")
#         return x, indices

class UpSample(torch.nn.Module):
    def __init__(self, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        if x.dim() == 4:
            x = x.squeeze(1)
        x = rearrange(x, "b l d-> b d l")
        x = F.upsample(x, scale_factor=self.scale_factor, mode="nearest")
        x = rearrange(x, "b d l -> b l d")
        return x

class MaxUnpool(torch.nn.Module):
    def __init__(self, kernel_size=2, stride=2):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x, indices):
        if x.dim() == 4:
            x = x.squeeze(1)
        x = rearrange(x, "b l d-> b d l")
        x = F.max_unpool1d(x, indices, self.kernel_size, self.stride)
        x = rearrange(x, "b d l -> b l d")
        return x
