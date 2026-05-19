import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.amp import GradScaler, autocast

import os
import argparse
from itertools import cycle
from dataclasses import dataclass

from model import Unet
from utils import EMA, set_seed

@dataclass
class Config:
    # Model
    base_dim: int = 64
    channel_mult: tuple = (1, 2, 4)
    time_emb_dim: int = 512
    channels: int = 1

    # Training
    max_steps: int = 16000
    batch_size: int = 256
    lr: float = 1e-4

    # Environment
    data_dir: str = '/kaggle/working/data'
    ckpt_dir: str = '/kaggle/working/checkpoints'
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed: int = 42

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
    set_seed(cfg.seed)

    model = Unet(cfg).to(cfg.device)
    ema_model = EMA(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scaler = GradScaler(cfg.device)

    loader = cycle(get_dataloader(cfg))

    for step in range(cfg.max_steps):
        x1, _ = next(loader)
        
        x1 = x1.to(cfg.device)
        x0 = torch.randn_like(x1)
        t = torch.rand(x1.shape[0], 1, 1, 1, dtype=x1.dtype ,device = cfg.device)

        x_t = t * x1 + (1 - t) * x0
        target = x1 - x0

        optimizer.zero_grad(set_to_none=True)

        with autocast(cfg.device):
            pred = model(x_t, t.squeeze())
            loss = F.mse_loss(pred, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        ema_model.update(step)

        if (step + 1) % (cfg.max_steps // 10) == 0:
            print(f'step {step + 1}: Loss - {loss.item()}')
    
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    torch.save({
        'model': model.state_dict(),
        'ema': ema_model.state_dict(),
    }, os.path.join(cfg.ckpt_dir, f"MNIST_{cfg.max_steps}steps.pt"))

def main():
    cfg = Config()

    parser = argparse.ArgumentParser(description='training configuration')
    parser.add_argument('-bs', '--batch_size', type=int, help='batch size of train data', metavar='', default=cfg.batch_size)
    parser.add_argument('-ms', '--max_steps', type=int, help='maximum steps for training', metavar='', default=cfg.max_steps)
    parser.add_argument('-lr', '--learning_rate', type=float, help='learning rate for the optimizer', metavar='', default=cfg.lr)
    parser.add_argument('-dr', '--data_dir', type=str, help='data folder location ', metavar='', default=cfg.data_dir)
    parser.add_argument('-cr', '--ckpt_dir', type=str, help='checkpoint folder location', metavar='', default=cfg.ckpt_dir)
    args = parser.parse_args()

    cfg = Config(
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        lr=args.learning_rate,
        data_dir=args.data_dir,
        ckpt_dir=args.ckpt_dir
    )

    train(cfg)

if __name__ == '__main__':
    main()