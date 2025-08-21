
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        Norm = lambda c: nn.InstanceNorm2d(c, affine=True, track_running_stats=False)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            Norm(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            Norm(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )
    def forward(self, x): return self.net(x)

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch//2, 2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # pad if needed
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffX//2, diffX - diffX//2,
                        diffY//2, diffY - diffY//2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, base=32, out_channels=3):
        super().__init__()
        self.inc = DoubleConv(in_channels, base)
        self.d1 = Down(base, base*2)
        self.d2 = Down(base*2, base*4)
        self.d3 = Down(base*4, base*8)
        self.d4 = Down(base*8, base*16)
        self.u1 = Up(base*16, base*8)
        self.u2 = Up(base*8, base*4)
        self.u3 = Up(base*4, base*2)
        self.u4 = Up(base*2, base)
        self.outc = nn.Conv2d(base, out_channels, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.d1(x1)
        x3 = self.d2(x2)
        x4 = self.d3(x3)
        x5 = self.d4(x4)
        x = self.u1(x5, x4)
        x = self.u2(x,  x3)
        x = self.u3(x,  x2)
        x = self.u4(x,  x1)
        return torch.sigmoid(self.outc(x))
