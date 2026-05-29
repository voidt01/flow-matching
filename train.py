import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast

import os
import yaml
import argparse
from itertools import cycle

from model import Unet
from config import Config
from data import get_dataloader_MNIST, get_dataloader_Simpsons
from samplers import sample_euler
from utils import EMA, save_samples, get_device, set_seed

def train(model, ema, optimizer, scaler, loader, device, cfg):

    os.makedirs(cfg.img_samples_dir, exist_ok=True)    
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    
    for step in range(cfg.max_steps):
        x1, _ = next(loader)
        
        x1 = x1.to(device)
        x0 = torch.randn_like(x1)
        t = torch.randn(x1.shape[0], 1, 1, 1, dtype=x1.dtype ,device=device)
        t = torch.sigmoid(t)

        x_t = t * x1 + (1 - t) * x0
        target = x1 - x0

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device):
            pred = model(x_t, t.squeeze())
            loss = F.mse_loss(pred, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        ema.update(step)

        if (step + 1) % (cfg.max_steps // 10) == 0:
            print(f'Step {step + 1}: Loss - {loss.item()}')
        
        if (step + 1) % cfg.log_img_every == 0:
            model_images = sample_euler(model, cfg, device, 12, 50)
            ema_images = sample_euler(ema.ema_model, cfg, device, 12, 50) # make sure ema_update_after_step < log_img_every
            save_samples(model_images, output_path=os.path.join(cfg.img_samples_dir, f"regular_model_{step} steps.png"))
            save_samples(ema_images, output_path=os.path.join(cfg.img_samples_dir, f"ema_model_{step} steps.png"))
                
    torch.save({
        'model': model.state_dict(),
        'ema': ema.state_dict(), # could be None if step < update_after_step
    }, os.path.join(cfg.ckpt_dir, f"{cfg.dataset}_{cfg.max_steps}steps.pt")) 

def main():
    parser = argparse.ArgumentParser(description='Training Configuration')
    parser.add_argument('-cf', '--config', type=str, metavar='', help='Path to YAML config file')
    args = parser.parse_args()

    try:
        cfg = Config.parse_yaml(args.config) if args.config else Config()
    except FileNotFoundError:
        print(f"config file not found: {args.config}")
        return
    except yaml.YAMLError as e:
        print(f"invalid YAML file:\n{e}")
        return
    except TypeError as e:
        print(f"invalid config fields:\n{e}")
        return

    set_seed(cfg.seed)
    device = get_device()

    model = Unet(
        base_dim=cfg.base_dim,
        channel_mult=cfg.channel_mult,
        channels=cfg.channels,
        time_emb_dim=cfg.time_emb_dim
    ).to(device)
    ema_model = EMA(model, update_after_step=cfg.ema_update_after_step, beta=cfg.ema_decay)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scaler = GradScaler(device=device)

    if cfg.dataset == 'mnist':
        loader = cycle(get_dataloader_MNIST(cfg.data_dir, cfg.batch_size))
    elif cfg.dataset == 'simpsons':
        loader = cycle(get_dataloader_Simpsons(cfg.data_dir, cfg.batch_size))
    else:
        raise ValueError('Unknown dataset')

    train(model, ema_model, optimizer, scaler, loader, device, cfg)

if __name__ == '__main__':
    main()