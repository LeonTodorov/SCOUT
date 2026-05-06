from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn


def _conv_bn_relu(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


def _make_block(in_ch: int, out_ch: int, num_convs: int, dropout: float = 0.0) -> nn.Sequential:
    layers: List[nn.Module] = []
    layers.append(_conv_bn_relu(in_ch, out_ch))
    for _ in range(num_convs - 1):
        layers.append(_conv_bn_relu(out_ch, out_ch))
    if dropout > 0:
        layers.append(nn.Dropout2d(dropout))
    return nn.Sequential(*layers)


class Decoder(nn.Module):
    """AdaFace embedding (512) -> image decoder producing BGR [-1,1]."""

    def __init__(self, output_size: int = 112):
        super().__init__()
        self.output_size = output_size

        if output_size % 7 != 0:
            raise ValueError(f"output_size must be divisible by 7, got {output_size}")
        ratio = output_size // 7
        if ratio & (ratio - 1) != 0:
            raise ValueError(f"output_size/7 must be power of 2, got {ratio}")
        self.num_upsamples = int(math.log2(ratio))

        self.fc = nn.Linear(512, 7 * 7 * 128)
        self.fc_bn = nn.BatchNorm1d(7 * 7 * 128)
        self.fc_act = nn.ReLU(inplace=True)
        self.fc_drop = nn.Dropout2d(0.5)

        # Base channel plan for 112x112
        plan: List[Tuple[int, int, float]] = [
            (512, 3, 0.5),
            (512, 3, 0.5),
            (256, 3, 0.5),
            (128, 2, 0.0),
        ]
        if self.num_upsamples > len(plan):
            # Extend with lightweight blocks for larger outputs
            extra = self.num_upsamples - len(plan)
            plan.extend([(128, 2, 0.0) for _ in range(extra)])
        elif self.num_upsamples < len(plan):
            plan = plan[: self.num_upsamples]

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        blocks: List[nn.Module] = []
        in_ch = 128
        for out_ch, num_convs, dropout in plan:
            blocks.append(self.upsample)
            blocks.append(_make_block(in_ch, out_ch, num_convs, dropout))
            in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)

        self.final = nn.Sequential(
            _conv_bn_relu(in_ch, 64),
            _conv_bn_relu(64, 64),
        )
        self.out = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        x = self.fc_act(self.fc_bn(self.fc(embedding)))
        x = x.view(-1, 128, 7, 7)
        x = self.fc_drop(x)
        x = self.blocks(x)
        x = self.final(x)
        x = self.out(x)
        return x
