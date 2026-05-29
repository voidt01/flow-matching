import torch

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