import math
from typing import List

import torch
import torch.nn as nn


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation:
      - AdaptiveAvgPool2d(1) -> (N,C,1,1)
      - Flatten to (N,C)
      - Linear C -> ceil(C/r) -> ReLU -> Linear -> Sigmoid
      - Reshape to (N,C,1,1) and multiply
    """

    def __init__(self, channels: int, reduction: int):
        super().__init__()
        hidden = max(1, int(math.ceil(channels / reduction)))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, hidden)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden, channels)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, _, _ = x.shape
        s = self.pool(x).view(n, c)          # (N,C)
        s = self.fc1(s)
        s = self.act(s)
        s = self.fc2(s)
        w = self.gate(s).view(n, c, 1, 1)    # (N,C,1,1)
        return x * w


class ConvBNReLU_SE(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, reduction: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.se = SEBlock(out_ch, reduction=reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.se(x)
        return x


class PetSE_CNN(nn.Module):
    """
    3 stages:
      Stage 1: blocks_per_stage blocks @ 64, then MaxPool2d(2)
      Stage 2: blocks_per_stage blocks @ 128, then MaxPool2d(2)
      Stage 3: blocks_per_stage blocks @ 256, then GlobalAvgPool
    Head: Dropout -> Linear(256->num_classes)
    """

    def __init__(
        self,
        num_classes: int = 37,
        reduction: int = 8,
        blocks_per_stage: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()
        if blocks_per_stage not in (2, 3):
            raise ValueError("blocks_per_stage must be 2 or 3")
        if reduction not in (8, 16):
            raise ValueError("reduction must be 8 or 16")

        self.stage1 = self._make_stage(in_ch=3, out_ch=64, n_blocks=blocks_per_stage, reduction=reduction)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.stage2 = self._make_stage(in_ch=64, out_ch=128, n_blocks=blocks_per_stage, reduction=reduction)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.stage3 = self._make_stage(in_ch=128, out_ch=256, n_blocks=blocks_per_stage, reduction=reduction)
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(256, num_classes)

    @staticmethod
    def _make_stage(in_ch: int, out_ch: int, n_blocks: int, reduction: int) -> nn.Sequential:
        layers: List[nn.Module] = []
        # first block changes channels
        layers.append(ConvBNReLU_SE(in_ch, out_ch, reduction=reduction))
        # remaining blocks keep channels
        for _ in range(n_blocks - 1):
            layers.append(ConvBNReLU_SE(out_ch, out_ch, reduction=reduction))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage1(x)
        x = self.pool1(x)

        x = self.stage2(x)
        x = self.pool2(x)

        x = self.stage3(x)
        x = self.gap(x).squeeze(-1).squeeze(-1)  # (N,256)

        x = self.dropout(x)
        x = self.fc(x)                           # (N,num_classes)
        return x
