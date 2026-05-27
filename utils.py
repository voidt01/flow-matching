import copy
import torch
import random
import numpy as np

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
    
def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False