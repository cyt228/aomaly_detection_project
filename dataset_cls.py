# dataset_cls.py
import os, csv, torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class PairDataset(Dataset):
    def __init__(self, orig_dir, recon_dir, split="train"):
        self.orig_dir = orig_dir
        self.recon_dir = recon_dir
        self.split = split

        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
        ])

        # no_defects.csv 在 orig_dir
        normal_file = os.path.join(orig_dir, "no_defects.csv")
        normals = set()
        if os.path.exists(normal_file):
            with open(normal_file, newline="") as f:
                reader = csv.reader(f)
                normals = {row[0] for row in reader}

        self.samples = []
        for f in os.listdir(orig_dir):
            if not f.lower().endswith((".png",".jpg",".jpeg")):
                continue
            orig_path = os.path.join(orig_dir, f)
            recon_path = os.path.join(recon_dir, f)
            if not os.path.exists(recon_path):
                continue
            label = 0 if f in normals else 1  # 0=normal, 1=defect
            self.samples.append((orig_path, recon_path, label))

    def __len__(self): 
        return len(self.samples)

    def __getitem__(self, idx):
        orig_path, recon_path, label = self.samples[idx]
        orig = self.transform(Image.open(orig_path).convert("RGB"))
        recon = self.transform(Image.open(recon_path).convert("RGB"))
        diff = torch.abs(orig - recon)      # [3,H,W]，值域仍在 [0,1]
        return diff, label
