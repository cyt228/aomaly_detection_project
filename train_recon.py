
import os, time, math, argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from dataset_recon import PairedImageFolder, make_split
from unet import UNet

# --- SSIM (simplified) ---
# window-based SSIM for 3-channel images in [0,1]
import torch.nn.functional as F

def _gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def _create_window(window_size, channel):
    _1D_window = _gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(img1, img2, window_size=11, size_average=True, val_range=1.0):
    # expects NCHW in [0,1]
    (_, channel, height, width) = img1.size()
    window = _create_window(window_size, channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = (0.01*val_range)**2
    C2 = (0.03*val_range)**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def psnr(yhat, y, eps=1e-8):
    mse = torch.mean((yhat - y) ** 2, dim=[1,2,3])
    return 20 * torch.log10(1.0 / torch.sqrt(mse + eps))

# --- Training ---
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input_dir', type=str, required=True, help='directory of defect images')
    ap.add_argument('--target_dir', type=str, required=True, help='directory of clean images (paired filenames)')
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--val_ratio', type=float, default=0.1, help='auto split if no val list provided')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', type=str, default='checkpoints_recon')
    ap.add_argument('--save_every', type=int, default=5)
    return ap.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # split
    train_files_path = os.path.join(args.out, 'train_files.txt')
    val_files_path   = os.path.join(args.out, 'val_files.txt')
    if os.path.exists(train_files_path) and os.path.exists(val_files_path):
        with open(train_files_path, 'r') as f:
            train_list = [l.strip() for l in f if l.strip()]
        with open(val_files_path, 'r') as f:
            val_list = [l.strip() for l in f if l.strip()]
    else:
        train_list, val_list = make_split(args.input_dir, seed=args.seed, val_ratio=args.val_ratio)
        with open(train_files_path, 'w') as f: f.write('\n'.join(train_list))
        with open(val_files_path, 'w')   as f: f.write('\n'.join(val_list))
        print(f"[split] train={len(train_list)} val={len(val_list)} (saved at {args.out})")

    # data
    train_ds = PairedImageFolder(args.input_dir, args.target_dir, file_list=train_list)
    val_ds   = PairedImageFolder(args.input_dir, args.target_dir, file_list=val_list)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_dl   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # model
    net = UNet().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    l1  = nn.L1Loss()

    best_val = float('inf')
    for epoch in range(1, args.epochs+1):
        net.train()
        tr_l1 = 0.0
        tr_ssim = 0.0
        n = 0
        for x, y, _ in train_dl:
            x = x.to(device)
            y = y.to(device)
            yhat = net(x)
            loss_l1 = l1(yhat, y)
            loss_ssim = 1 - ssim(yhat, y)   # maximize ssim
            loss = loss_l1 + loss_ssim

            opt.zero_grad()
            loss.backward()
            opt.step()

            bs = x.size(0)
            tr_l1 += loss_l1.item() * bs
            tr_ssim += (1 - loss_ssim.item()) * bs
            n += bs

        tr_l1 /= max(1, n)
        tr_ssim /= max(1, n)

        # validation
        net.eval()
        va_l1 = 0.0
        va_ssim = 0.0
        n = 0
        with torch.no_grad():
            for x, y, names in val_dl:
                x = x.to(device); y = y.to(device)
                yhat = net(x)
                loss_l1 = l1(yhat, y)
                va_l1 += loss_l1.item()*x.size(0)
                va_ssim += ssim(yhat, y).item()*x.size(0)
                n += x.size(0)

        va_l1 /= max(1, n)
        va_ssim /= max(1, n)

        print(f"Epoch {epoch:03d}/{args.epochs} | train L1={tr_l1:.4f} SSIM={tr_ssim:.4f} | val L1={va_l1:.4f} SSIM={va_ssim:.4f}")

        # save samples and ckpt
        if epoch % args.save_every == 0 or epoch == args.epochs:
            sample_dir = os.path.join(args.out, f"samples_ep{epoch:03d}")
            os.makedirs(sample_dir, exist_ok=True)
            # dump first batch of val
            with torch.no_grad():
                for x, y, names in val_dl:
                    x = x.to(device); y = y.to(device)
                    yhat = net(x)
                    for i in range(x.size(0)):
                        grid = torch.stack([x[i], yhat[i].clamp(0,1), y[i]], dim=0)  # [in, recon, target]
                        save_image(grid, os.path.join(sample_dir, f"{names[i]}"))
                    break

            ckpt_path = os.path.join(args.out, f"recon_unet_ep{epoch:03d}.pt")
            torch.save({'model': net.state_dict(), 'epoch': epoch}, ckpt_path)

        # track best
        if va_l1 < best_val:
            best_val = va_l1
            torch.save({'model': net.state_dict(), 'epoch': epoch}, os.path.join(args.out, "recon_unet_best.pt"))

if __name__ == "__main__":
    main()
