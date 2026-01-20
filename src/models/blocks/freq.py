import torch
import torch.nn as nn
import torch.nn.functional as F

class FreqEnhance(nn.Module):
    """
    Simple frequency enhancement:
    - Compute FFT amplitude
    - Project to freq_ch and inject back as attention-like map
    """
    def __init__(self, in_ch, freq_ch=16):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, freq_ch, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(freq_ch, in_ch, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (B,C,H,W)
        fft = torch.fft.rfft2(x, norm="ortho")
        amp = torch.abs(fft)
        amp = F.interpolate(amp, size=x.shape[-2:], mode="bilinear", align_corners=False)
        gate = self.proj(amp)
        return x * (1.0 + gate)

