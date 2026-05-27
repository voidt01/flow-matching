import torch

import os
import yaml
import argparse
import matplotlib.pyplot as plt

from model import Unet
from train import Config
from utils import get_device

@torch.no_grad()
def sample_euler(
    model,
    cfg,
    device,
    n_samples: int,
    num_steps: int,
    fixed_noise = None 
):
    if fixed_noise is not None:
        x = fixed_noise.to(device)
    else:
        x = torch.randn(n_samples, cfg.channels, cfg.img_size, cfg.img_size, device=device)

    t = torch.linspace(0, 1, steps=num_steps, device=device)
    dt = 1 / num_steps

    for i in range(num_steps):
        t_val = torch.full((x.shape[0],), t[i], device=device)
        x = x + dt * model(x, t_val)
    
    return x

def save_samples(images, output_path='samples.png'):
    # denormalize
    images = (images + 1) / 2
    images = images.clamp(0, 1)

    fig, axes = plt.subplots(1, len(images))
    for i, ax in enumerate(axes):
        ax.imshow(images[i].detach().cpu().squeeze().numpy(), cmap='gray')
        ax.axis('off')
     
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Sampling Configuration')
    parser.add_argument('-cf', '--config', type=str, metavar='', help='Path to YAML config file')
    parser.add_argument('-fn', '--fixed_noise', type=str, metavar='', help='Path to fixed gaussian noise')
    parser.add_argument('-md', '--model', type=str, metavar='', help='Checkpoint to use for sample (model/ema)')
    parser.add_argument('-ns', '--n_samples', type=int, metavar='', default=10, help='Total of images to be sample to')
    parser.add_argument('-ss', '--sampling_steps', type=int, metavar='', default=50, help='How many steps to create a sample image')
    parser.add_argument('-op', '--output_path', type=str, metavar='', help='Path to save sampled images')
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

    device = get_device()

    noise = torch.load(args.fixed_noise) if args.fixed_noise else None
    ckpt_path = os.path.join(cfg.ckpt_dir, f"{cfg.dataset}_{cfg.max_steps}steps.pt")
    ckpt = torch.load(ckpt_path, map_location=device)

    model = Unet(
        base_dim=cfg.base_dim,
        channel_mult=cfg.channel_mult,
        channels=cfg.channels,
        time_emb_dim=cfg.time_emb_dim
    ).to(device)
    model.load_state_dict(ckpt[args.model])
    model.eval()

    images = sample_euler(model, cfg, n_samples=args.n_samples, num_steps=args.sampling_steps, fixed_noise=noise, device=device)
    save_samples(images, args.output_path)

if __name__ == '__main__':
    main()