
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from typing import List, Tuple
import random

def default_transform(size=512):
    return T.Compose([
        T.ToTensor(),                       # [0,1]
    ])

class PairedImageFolder(Dataset):
    """
    Minimal paired dataset for reconstruction.
    Expects two directories with matching filenames:
      input_dir/:  defect images (x)
      target_dir/: clean images  (y)
    """
    def __init__(self, input_dir: str, target_dir: str, file_list: List[str]=None, transform=None):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.files = sorted(file_list) if file_list else sorted([
            f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        self.transform = transform or default_transform()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        x_path = os.path.join(self.input_dir, fname)
        y_path = os.path.join(self.target_dir, fname)
        x = Image.open(x_path).convert('RGB')
        y = Image.open(y_path).convert('RGB')
        if self.transform:
            x = self.transform(x)
            y = self.transform(y)
        return x, y, fname

def make_split(input_dir: str, seed: int=42, val_ratio: float=0.1) -> Tuple[List[str], List[str]]:
    """Create a deterministic filename split based on sorted list + seed."""
    files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))])
    rng = random.Random(seed)
    idxs = list(range(len(files)))
    rng.shuffle(idxs)
    n_val = max(1, int(len(files)*val_ratio)) if len(files) > 0 else 0
    val_idxs = set(idxs[:n_val])
    train_files = [files[i] for i in range(len(files)) if i not in val_idxs]
    val_files   = [files[i] for i in range(len(files)) if i in val_idxs]
    return train_files, val_files
