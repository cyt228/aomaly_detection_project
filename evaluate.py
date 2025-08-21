# evaluate.py
import argparse, torch
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from dataset_cls import PairDataset
import torch.nn as nn

def build_model(num_classes=2):
    net = models.resnet18(weights=None)
    net.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
    net.fc = nn.Linear(net.fc.in_features, num_classes)
    return net

@torch.no_grad()
def evaluate(orig_dir, recon_dir, ckpt, batch_size=32):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = PairDataset(orig_dir, recon_dir, split="test")
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    net = build_model()
    net.load_state_dict(torch.load(ckpt, map_location=device))
    net.to(device).eval()

    y_true, y_pred = [],[]
    for x,y in loader:
        x = x.to(device)
        out = net(x)
        pred = out.argmax(1).cpu().numpy()
        y_pred.extend(pred)
        y_true.extend(y.numpy())

    acc = accuracy_score(y_true,y_pred)
    f1 = f1_score(y_true,y_pred,average="macro")
    cm = confusion_matrix(y_true,y_pred)
    print(f"Accuracy={acc:.4f} | F1={f1:.4f}")
    print("Confusion Matrix:\n", cm)
    print("\nClassification Report:\n", classification_report(y_true,y_pred,target_names=["normal","defect"]))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig_dir", required=True)
    ap.add_argument("--recon_dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()
    evaluate(args.orig_dir, args.recon_dir, args.ckpt, args.batch_size)
