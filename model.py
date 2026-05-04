import math 
from functools import partial

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch import einsum

from einops import reduce, rearrange
from einops.layers.torch import Rearrange

# U-net
class Unet(nn.Module):
    def __init__(self, cfg):
        dims = [cfg.base_dim * channel for channel in cfg.channel_mult]
        in_out = list(zip(dims[:-1], dims[1:]))

        self.time_emb = TimeEmbedding(cfg.time_emb_dim)
        self.init_conv = nn.Conv2d(cfg.channels, dims[0], 7, padding=3)

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind == len(in_out) - 1
            self.downs.append(nn.ModuleList([
                ResnetBlock(dim_in, dim_in, time_emb_dim=cfg.time_emb_dim),
                ResnetBlock(dim_in, dim_in, time_emb_dim=cfg.time_emb_dim),
                downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1)
            ]))  
                
        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=cfg.time_emb_dim)
        self.mid_attn = Attention(mid_dim)
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=cfg.time_emb_dim)        

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == len(in_out) - 1
            self.ups.append(nn.ModuleList([
                ResnetBlock(dim_out + dim_in, dim_out, time_emb_dim=cfg.time_emb_dim),
                ResnetBlock(dim_out + dim_in, dim_out, time_emb_dim=cfg.time_emb_dim),
                upsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1)
            ]))

        self.final_res = ResnetBlock(dims[0] * 2, dims[0], time_emb_dim=cfg.time_emb_dim)
        self.final_conv = nn.Conv2d(dims[0], cfg.channels, 1)

    def forward(self, x, t):
        t = self.time_emb(t)
        x = self.init_conv(x)
        r = x.clone()

        skips = []
        for block1, block2, down in self.downs:
            x = block1(x, t)
            skips.append(x)
            x = block2(x, t)
            skips.append(x)

            x = down(x)
        
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, up in self.ups:
            x = torch.cat([x, skips.pop()], dim=1)
            x = block1(x, t)
            x = torch.cat([x, skips.pop()], dim=1)
            x = block2(x, t)

            x = up(x)
        
        x = torch.cat([x, r], dim=1)
        x = self.final_res(x, t)
        return self.final_conv(x)


## Building Blocks
class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()

        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dims = self.heads * dim_head

        self.to_qkv = nn.Conv2d(dim, hidden_dims * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dims, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv
        )
        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum('b h i j, b h d j -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)

        return self.to_out(out)

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None):
        super().__init__()

        self.time_proj = (
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, dim_out * 2)
            )
            if exists(time_emb_dim)
            else None
        )

        if exists(self.time_proj):
            nn.init.zeros_(self.time_proj[-1].weight)
            nn.init.zeros_(self.time_proj[-1].bias)

        self.block1 = Block(dim, dim_out)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None

        if exists(self.time_proj) and exists(time_emb):
            time = self.time_proj(time_emb)
            time = rearrange(time, 'b c -> b c 1 1')
            scale_shift = time.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)

        return h + self.res_conv(x)

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
        mean = reduce(weight, 'o ... -> o 1 1 1', 'mean')
        variance = reduce(weight, 'o ... -> o 1 1 1', partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (variance + eps).rsqrt()

        return F.conv2d(
            x,
            normalized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups
        )
    
## Time Embedding
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        self.dim = dim
        self.proj = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, t):
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device = t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return self.proj(emb)

## Helpers 
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

