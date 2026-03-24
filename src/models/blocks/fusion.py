import torch
import torch.nn as nn


class BottleneckFusion(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(ch * 2, ch, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1),
        )

    def forward(self, a, b):
        return self.fuse(torch.cat([a, b], dim=1))