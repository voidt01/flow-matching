import yaml
from dataclasses import dataclass, field

@dataclass
class Config:
    # Model
    base_dim: int = 64
    channel_mult: list = field(default_factory=lambda: [1, 2, 2, 2])
    time_emb_dim: int = 512

    # Training
    max_steps: int = 15000
    batch_size: int = 128
    max_lr: float = 2e-4
    min_lr: float = 3e-5
    seed: int = 42
    ema_update_after_step: int = 1000
    ema_decay: float = 0.9995

    # Dataset
    dataset: str = 'mnist'
    channels: int = 1
    img_size: int = 28

    # Environment
    data_dir: str = './data'
    ckpt_dir: str = './checkpoints'
    img_samples_dir: str = './img-samples'

    # Logging
    log_img_every: int = 2000

    @classmethod
    def parse_yaml(cls, path):
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)
