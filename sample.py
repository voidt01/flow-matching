import torch
import argparse
import matplotlib.pyplot as plt

from model import Unet
from train import Config

@torch.no_grad()
def sample_euler(
    model,
    cfg,
    n_samples: int,
    num_steps: int,
    fixed_noise = None 
):
    if fixed_noise is not None:
        x = fixed_noise.to(cfg.device)
    else:
        x = torch.randn(n_samples, cfg.channels, 28, 28, device=cfg.device)

    t = torch.linspace(0, 1, steps=num_steps, device=cfg.device)
    dt = 1 / num_steps

    for i in range(num_steps):
        t_val = torch.full((x.shape[0],), t[i], device=cfg.device)
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
    cfg = Config()

    parser = argparse.ArgumentParser(description='Sampling Configuration')
    parser.add_argument('-cp', '--ckpt_path', type=str, required=True, metavar='', help='path to checkpoint file')
    parser.add_argument('-md', '--model', type=str, required=True, metavar='', help='which model for sampling (temporary for learning)')
    parser.add_argument('-op', '--output_path', type=str, metavar='', help='path to save sampled image')
    parser.add_argument('-ns', '--n_samples', type=int, metavar='', default=3, help='n sample image to save')
    parser.add_argument('-ss', '--sampling_steps', type=int, metavar='', default=100, help='n steps for sampling')
    parser.add_argument('-fn', '--fixed_noise', type=str, metavar='', default=None, help='path for fixed noise tensor')
    args = parser.parse_args()

    noise = torch.load(args.fixed_noise) if args.fixed_noise else None
    ckpt = torch.load(args.ckpt_path, map_location=cfg.device)

    model = Unet(cfg).to(cfg.device)
    model.load_state_dict(ckpt[args.model])
    model.eval()

    images = sample_euler(model, cfg, n_samples=args.n_samples, num_steps=args.sampling_steps, fixed_noise=noise)
    save_samples(images, args.output_path)

if __name__ == '__main__':
    main()