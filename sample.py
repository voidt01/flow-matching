import torch

import os
import yaml
import argparse

from model import Unet
from train import Config
from utils import save_samples, get_device
from samplers import sample_euler, sample_heun

def main():
    parser = argparse.ArgumentParser(description='Sampling Configuration')
    parser.add_argument('-cf', '--config', type=str, metavar='', help='Path to YAML config file')
    parser.add_argument('-fn', '--fixed_noise', type=str, metavar='', help='Path to fixed gaussian noise')
    parser.add_argument('-md', '--model', type=str, metavar='', help='Checkpoint to use for sample (model/ema)')
    parser.add_argument('-sm', '--sampler', type=str, metavar='', help='ODE solver for sampling (euler/heun)')
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

    if args.sampler == 'euler':
        images = sample_euler(model, cfg, n_samples=args.n_samples, num_steps=args.sampling_steps, fixed_noise=noise, device=device)
    elif args.sampler == 'heun':
        images = sample_heun(model, cfg, n_samples=args.n_samples, num_steps=args.sampling_steps, fixed_noise=noise, device=device)
    else:
        raise ValueError('Unknown sampler')
    
    save_samples(images, args.output_path)

if __name__ == '__main__':
    main()