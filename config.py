import torch
from dataclasses import dataclass

@dataclass
class Config:
    # Model
    base_dim: int = 64
    channel_mult: tuple = (1, 2, 2, 2)
    time_emb_dim: int = 512
    channels: int = 1

    # Training
    max_steps: int = 15000
    batch_size: int = 256
    lr: float = 1e-4

    # Environment
    data_dir: str = '/kaggle/working/data'
    ckpt_dir: str = '/kaggle/working/checkpoints'
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed: int = 42
    dataset: str = 'mnist'

    @property
    def dims(self):
        return [self.base_dim * channel for channel in self.channel_mult]