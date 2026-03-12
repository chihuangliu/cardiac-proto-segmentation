"""
src/models/encoder.py
HierarchicalEncoder2D: multi-scale backbone exposing 4 feature maps for prototype attachment.

Input:  (B, 1, 256, 256)
Output: {1: Z_1(B,32,128,128), 2: Z_2(B,64,64,64),
          3: Z_3(B,128,32,32),  4: Z_4(B,256,16,16)}

Each level: Conv2d(stride=2) → BN → ReLU → ResBlock
"""

import torch
import torch.nn as nn


# ── Building blocks ───────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    """Standard residual block: two 3×3 convs with BN+ReLU, identity shortcut.
    When in_ch != out_ch a 1×1 projection is used for the shortcut.
    No internal stride change — striding is done before this block.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.shortcut = (
            nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                          nn.BatchNorm2d(out_ch))
            if in_ch != out_ch else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x) + self.shortcut(x))


class EncoderBlock(nn.Module):
    """One hierarchical encoder level.
    Structure: Conv2d(stride=2) → BN → ReLU → ResBlock
    Downsamples spatial dims by 2 and maps to out_ch.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.res = ResBlock(out_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.res(self.down(x))


# ── Main backbone ─────────────────────────────────────────────────────────────

class HierarchicalEncoder2D(nn.Module):
    """
    Multi-scale 2D encoder for prototype segmentation.

    Input:  (B, 1, 256, 256)
    Outputs (returned as dict keyed by level index 1–4):
      Z_1  (B,  32, 128, 128)   fine texture / boundary
      Z_2  (B,  64,  64,  64)   local edge / inter-structure
      Z_3  (B, 128,  32,  32)   structure-level context
      Z_4  (B, 256,  16,  16)   global cardiac layout
    """

    CHANNELS = {1: 32, 2: 64, 3: 128, 4: 256}

    def __init__(self, in_channels: int = 1):
        super().__init__()
        ch = self.CHANNELS
        self.level1 = EncoderBlock(in_channels, ch[1])   # 256→128
        self.level2 = EncoderBlock(ch[1], ch[2])          # 128→64
        self.level3 = EncoderBlock(ch[2], ch[3])          # 64→32
        self.level4 = EncoderBlock(ch[3], ch[4])          # 32→16

    def forward(self, x: torch.Tensor) -> dict[int, torch.Tensor]:
        z1 = self.level1(x)
        z2 = self.level2(z1)
        z3 = self.level3(z2)
        z4 = self.level4(z3)
        return {1: z1, 2: z2, 3: z3, 4: z4}

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
