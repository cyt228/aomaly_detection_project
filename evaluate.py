# evaluate.py
import argparse, torch
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from dataset_cls import PairDataset
import torch.nn as nn

def build_model(num_classes=2):
    # 3ch 差分輸入
    net = models.resnet18(weights=None)  # 評估只載你自己的 ckpt
    # 第一層保持 3ch，不改
    net.fc = nn.Linear(net.fc.in_features, num_classes)
    return net

@torch.no_grad()
def evaluate(orig_dir, recon_dir, ckpt, batch_size=32):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = PairDataset(orig_dir, recon_dir, split="test")
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)

    net = build_model()
    state = torch.load(ckpt, map_location=device)
    net.load_state_dict(state)  # 需與 3ch 模型相符
    net.to(device).eval()

    y_true, y_pred = [], []
    for x, y in loader:
        # x 目前是 6ch: [orig(3), recon(3)]，在線轉成 3ch 差分
        if x.size(1) == 6:
            orig  = x[:, 0:3, ...]
            recon = x[:, 3:6, ...]
            x = torch.abs(orig - recon)  # 3ch diff
        # 若你的 Dataset 已經直接回傳 3ch diff，這裡 size(1) 就是 3，會直接沿用

        x = x.to(device)
        out = net(x)
        pred = out.argmax(1).cpu().numpy()
        y_pred.extend(pred)
        # y 可能是 tensor，轉成 list 以保險
        y_true.extend(y.cpu().numpy() if torch.is_tensor(y) else y)

    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="macro")
    cm  = confusion_matrix(y_true, y_pred)
    print(f"Accuracy={acc:.4f} | F1={f1:.4f}")
    print("Confusion Matrix:\n", cm)
    print("\nClassification Report:\n",
          classification_report(y_true, y_pred, target_names=["normal","defect"]))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig_dir", required=True)
    ap.add_argument("--recon_dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()
    evaluate(args.orig_dir, args.recon_dir, args.ckpt, args.batch_size)
