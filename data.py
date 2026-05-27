from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

class UnlabeledImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        valid_extension = {'.png', '.jpg', '.jpeg'}
        self.image_paths = [
            p for p in self.root_dir.rglob('*')
            if p.suffix.lower() in valid_extension
        ]
    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        
        return img, 0 # matches how MNIST return single data

def get_dataloader_Simpsons(data_dir, batch_size):
    transform = v2.Compose([
        v2.CenterCrop(200),
        v2.Resize((64, 64), interpolation=v2.InterpolationMode.BICUBIC, antialias=True),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = UnlabeledImageDataset(
        root_dir=data_dir,
        transform=transform
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

def get_dataloader_MNIST(data_dir, batch_size):
    transform = v2.Compose([
        v2.ToImage(),
        v2.Normalize((0.5,), (0.5,))
    ])

    dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
