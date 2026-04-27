import math 
from functools import partial

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch import einsum

from einops import reduce, rearrage
from einops.layers.torch import Rearrange


class Block(nn.Module):
    def __init__(self, dim, dim_out):
        super().__init__()

        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(num_groups=8, num_channels=dim_out)
        self.act = nn.SiLU()
    
    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (1 + scale) + shift
        
        x = self.act(x)
        return x

class WeightStandardizedConv2d(nn.Conv2d):
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, 'o ... -> b 1 1 1', 'mean')
        variance = reduce(weight, 'o ... -> b 1 1 1', partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (variance + eps).rsqrt()

        return F.conv2d(
            weight=normalized_weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,

        )

def downsample(dim, dim_out=None):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1=2, p2=2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1)
    )

def upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1) 
    )

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d
