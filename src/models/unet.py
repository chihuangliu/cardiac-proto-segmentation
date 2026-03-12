"""
src/models/unet.py
Standard 2D U-Net for multi-class cardiac segmentation baseline.
Input:  (B, 1, 256, 256)
Output: (B, 8, 256, 256) logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Two Conv2d → BN → ReLU layers."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    """MaxPool2d(2) → ConvBlock."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    """Bilinear upsample × 2 → concat skip → ConvBlock."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Handle size mismatch from odd dimensions
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class UNet2D(nn.Module):
    """
    Standard 2D U-Net.
    Encoder channels: [32, 64, 128, 256]
    Bottleneck: 512
    Decoder mirrors encoder.
    """

    def __init__(self, in_channels: int = 1, n_classes: int = 8, base_ch: int = 32):
        super().__init__()
        ch = [base_ch, base_ch * 2, base_ch * 4, base_ch * 8]  # [32, 64, 128, 256]
        btn = base_ch * 16  # 512

        # Encoder
        self.enc1 = ConvBlock(in_channels, ch[0])   # (B,32,256,256)
        self.enc2 = DownBlock(ch[0], ch[1])          # (B,64,128,128)
        self.enc3 = DownBlock(ch[1], ch[2])          # (B,128,64,64)
        self.enc4 = DownBlock(ch[2], ch[3])          # (B,256,32,32)

        # Bottleneck
        self.bottleneck = DownBlock(ch[3], btn)      # (B,512,16,16)

        # Decoder
        self.dec4 = UpBlock(btn, ch[3], ch[3])       # (B,256,32,32)
        self.dec3 = UpBlock(ch[3], ch[2], ch[2])     # (B,128,64,64)
        self.dec2 = UpBlock(ch[2], ch[1], ch[1])     # (B,64,128,128)
        self.dec1 = UpBlock(ch[1], ch[0], ch[0])     # (B,32,256,256)

        # Output
        self.head = nn.Conv2d(ch[0], n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)
        b = self.bottleneck(s4)
        d4 = self.dec4(b, s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)
        return self.head(d1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
