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
        n_samples = x.shape[0]
    else:
        x = torch.randn(n_samples, cfg.channels, cfg.img_size, cfg.img_size, device=device)

    t = torch.linspace(0, 1, steps=num_steps + 1, device=device)

    for i in range(num_steps):
        dt = t[i + 1] - t[i]

        t_val = torch.full((n_samples, ), t[i], device=device)
        x = x + dt * model(x, t_val)
    
    return x

@torch.no_grad()
def sample_heun(
    model,
    cfg,
    device,
    n_samples: int,
    num_steps: int,
    fixed_noise = None     
):
    if fixed_noise is not None:
        x = fixed_noise.to(device)
        n_samples = x.shape[0]
    else:
        x = torch.randn(n_samples, cfg.channels, cfg.img_size, cfg.img_size, device=device)
    
    t = torch.linspace(0, 1, steps=num_steps + 1, device=device)

    for i in range(num_steps):
        dt = t[i + 1] - t[i]

        t_curr = torch.full((n_samples, ), t[i], device=device)
        t_next = torch.full((n_samples, ), t[i + 1], device=device)
        
        k1 = model(x, t_curr)
        x_pred = x + dt * k1
        k2 = model(x_pred, t_next)

        x = x + 0.5 * dt * (k1 + k2)

    return x