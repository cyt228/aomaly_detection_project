
import os, argparse, torch
from torchvision.utils import save_image
from PIL import Image
from dataset_recon import default_transform
from unet import UNet

def to_tensor(img):
    return default_transform()(img)

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input_dir', required=True, help='directory of defect images')
    ap.add_argument('--ckpt', required=True, help='path to recon_unet_*.pt')
    ap.add_argument('--out_dir', default='recon_out')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = UNet().to(device)
    state = torch.load(args.ckpt, map_location=device)
    net.load_state_dict(state['model'])
    net.eval()

    files = [f for f in sorted(os.listdir(args.input_dir)) if f.lower().endswith(('.png','.jpg','.jpeg'))]
    for f in files:
        img = Image.open(os.path.join(args.input_dir, f)).convert('RGB')
        x = to_tensor(img).unsqueeze(0).to(device)
        yhat = net(x).clamp(0,1)
        save_image(yhat[0], os.path.join(args.out_dir, f))

if __name__ == "__main__":
    main()
