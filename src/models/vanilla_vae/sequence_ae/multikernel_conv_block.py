from torch import nn
import torch
import math
from einops import rearrange
from torch.nn.init import trunc_normal_


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


class MultiKernelConvBlock(nn.Module):
    def __init__(
        self,
        dim=10,
        dim_out=None,
        kernel_sizes=[1, 3, 5],
        gelu=True,
        norm_type="group",
        linear_head=False,
        dilation=1,
        transpose=False
    ):
        super().__init__()
        self.kernel_sizes = [
            self.toOdd(kernel_size) for kernel_size in kernel_sizes
        ]
        dim_out = default(dim_out, dim)
        original_dim_out = dim_out
        dim_out = dim_out if linear_head else dim_out // len(kernel_sizes)
        if (not linear_head) and original_dim_out % len(kernel_sizes) != 0:
            raise ValueError(
                "dim_out must be divisible by the number of kernel sizes"
            )
        if transpose is False:
            self.conv_layers = nn.ModuleList(
                [
                    nn.Conv1d(
                        dim,
                        dim_out,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2 if dilation == 1 else dilation,
                        groups=1,
                        dilation=dilation,
                    )
                    for kernel_size in self.kernel_sizes
                ]
            )
        else:
            self.conv_layers = nn.ModuleList(
                [
                    nn.ConvTranspose1d(
                        dim,
                        dim_out,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2 if dilation == 1 else dilation,
                        groups=1,
                        dilation=dilation,
                    )
                    for kernel_size in self.kernel_sizes
                ]
            )
        # self.norm is a nn.GroupNorm if norm_type is group
        # and self.norm is nn.batchnorm1d if norm_type is batch
        # and self.norm is None if norm_type is None
        self.norm = None
        if norm_type == "group":
            self.norm = nn.GroupNorm(dim, dim)
        elif norm_type == "batch":
            self.norm = nn.BatchNorm1d(dim)
        self.gelu = nn.GELU() if gelu else None
        self.linear_head = (
            nn.Linear(len(kernel_sizes) * dim_out, dim_out)
            if linear_head
            else None
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, torch.nn.Linear) and m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, torch.nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, torch.nn.Conv1d):
            fan_out = m.kernel_size[0] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Sequential):
            for submodule in m.children():
                self._init_weights(submodule)
        elif isinstance(m, nn.ModuleList):
            for submodule in m:
                self._init_weights(submodule)

    def toOdd(self, num):
        if num % 2 == 0:
            return num + 1
        else:
            return num

    def forward(self, x):
        # fix: x.shape = (batch, seq_len, dim) if x.shape = (batch, 1, dim, seq_len)
        if x.dim() == 4:
            x = x.squeeze(1)

        x = rearrange(x, "b s d -> b d s")
        if exists(self.norm):
            x = self.norm(x)
        if exists(self.gelu):
            x = self.gelu(x)
        x = torch.cat([conv(x) for conv in self.conv_layers], dim=1)
        x = rearrange(x, "b d s -> b s d")
        if exists(self.linear_head):
            # rearrange x from (batch, dim, seq_len) to (batch, seq_len, dim)
            x = self.linear_head(x)
        return x
