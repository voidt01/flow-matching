import copy
import math
import torch
import random
import numpy as np
import matplotlib.pyplot as plt

class EMA:
    def __init__(self, model, update_after_step, beta):
        self.beta = beta
        self.update_after_step = update_after_step
        self.model = model
        self.ema_model = None

    @torch.no_grad()
    def update(self, step):
        if step < self.update_after_step:
            return
        
        self.init_after_step(step)

        for m_param, ema_param in zip(self.model.parameters(), self.ema_model.parameters()):
            ema_param.mul_(self.beta).add_(m_param * (1 - self.beta))

    def init_after_step(self, step):
        if step == self.update_after_step:
            self.ema_model = copy.deepcopy(self.model).eval()
            for p in self.ema_model.parameters():
                p.requires_grad_(False)
    
    def state_dict(self):
        if self.ema_model is None:
            return None
        return self.ema_model.state_dict()
    
    def load_state_dict(self, state_dict):
        if state_dict is None:
            return
        self.ema_model = copy.deepcopy(self.model).eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)
        self.ema_model.load_state_dict(state_dict)
    
def save_samples(images, output_path='samples.png'):
    if len(images.shape) == 3:
        images = images.unsqueeze(0) # [C, H, W] -> [1, C, H, W]

    # denormalize [-1, 1] -> [0, 1]
    images = (images + 1) / 2
    images = images.clamp(0, 1)

    num_img = len(images)    

    n_cols = int(math.ceil(math.sqrt(num_img)))
    n_rows = int(math.ceil(num_img / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.5, n_rows * 2.5), squeeze=False) # return 2d grid
    axes_flat = axes.flatten() # flatten to 1d for single loop

    for i in range(len(axes_flat)):
        ax = axes_flat[i]
        
        if i < num_img:
            img = images[i].detach().cpu()

            if img.shape[0] == 1:
                ax.imshow(img.squeeze().numpy(), cmap='gray')
            else:
                ax.imshow(img.permute(1, 2, 0).numpy())
        else:
            ax.axis('off')
            continue
        
        ax.axis('off')
     
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

def cycle(iterable):
    """
    generator for infinite stream data (step-based training)
    """
    while True:
        for batch in iterable:
            yield batch

def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False