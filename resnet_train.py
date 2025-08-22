# train.py
import argparse, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from dataset_cls import PairDataset

def build_model(num_classes=2):
    net = models.resnet18(weights="IMAGENET1K_V1") 
    net.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    net.fc = nn.Linear(net.fc.in_features, num_classes)
    return net

def train(orig_dir, recon_dir, out_ckpt="cls_resnet18.pt", epochs=20, lr=1e-4, batch_size=32):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = PairDataset(orig_dir, recon_dir, split="train")
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2)

    net = build_model().to(device)
    crit = nn.CrossEntropyLoss()
    opt = optim.Adam(net.parameters(), lr=lr)

    for ep in range(1, epochs+1):
        net.train()
        run_loss, correct, total = 0,0,0
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            opt.zero_grad()
            out = net(x)
            loss = crit(out,y)
            loss.backward()
            opt.step()
            run_loss += loss.item()*x.size(0)
            correct += (out.argmax(1)==y).sum().item()
            total += y.size(0)
        print(f"Epoch {ep}/{epochs} | Loss {run_loss/total:.4f} | Acc {correct/total:.4f}")

    torch.save(net.state_dict(), out_ckpt)
    print(f"[OK] saved model to {out_ckpt}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig_dir", required=True)
    ap.add_argument("--recon_dir", required=True)
    ap.add_argument("--out_ckpt", default="cls_resnet18.pt")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()
    train(args.orig_dir, args.recon_dir, args.out_ckpt, args.epochs, args.lr, args.batch_size)
