import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.amp import GradScaler, autocast

import os
from itertools import cycle
from dataclasses import dataclass
from einops import rearrange

from model import Unet

@dataclass
class TrainConfig:
    # Model
    base_dim: int = 64
    channel_mult: tuple = (1, 2, 4)
    time_emb_dim: int = 512
    channels: int = 1

    # Training
    max_steps: int = 16000
    batch_size: int = 128
    lr: float = 1e-4

    # Environment
    data_dir: str = 'kaggle/working/data'
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    @property
    def dims(self):
        return [self.base_dim * channel for channel in self.channel_mult]

def get_dataloader(cfg):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = datasets.MNIST(
        root=cfg.data_dir,
        train=True,
        download=True,
        transform=transform
    )

    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

def train(cfg):
    model = Unet(cfg).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scaler = GradScaler(cfg.device)
    loader = cycle(get_dataloader(cfg))

    print('currently run training')
    for step in range(cfg.max_steps):
        x1, _ = next(loader)
        
        x1 = x1.to(cfg.device)
        x0 = torch.randn_like(x1)
        t = torch.rand(cfg.batch_size, dtype=x1.dtype ,device = cfg.device)
        t_mod = rearrange(t, 'b -> b 1 1 1')

        x_t = t_mod * x1 + (1 - t_mod) * x0
        target = x1 - x0

        optimizer.zero_grad(set_to_none=True)

        with autocast(cfg.device):
            pred = model(x_t, t)
            loss = F.mse_loss(pred, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if (step + 1) % (cfg.max_steps // 10) == 0:
            print(f'step {step + 1}: Loss - {loss.item()}')

def main():
    cfg = TrainConfig()
    train(cfg)

if __name__ == '__main__':
    main()