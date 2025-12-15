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
        """
        x: (B, C, H, W) - có thể fp16/bf16 do AMP
        Trả về: (B, C, H, W) cùng dtype với x
        """
        orig_dtype = x.dtype

        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            x32 = x.float()  # (B,C,H,W) float32
            fft = torch.fft.rfft2(x32, norm="ortho")   
            amp = torch.abs(fft)                       

            amp = F.interpolate(
                amp,
                size=x32.shape[-2:],                   
                mode="bilinear",
                align_corners=False,
            )

            gate32 = self.proj(amp)                    
            gate32 = torch.nan_to_num(gate32, nan=0.0, posinf=1e6, neginf=-1e6)

        gate = gate32.to(dtype=orig_dtype)
        return x * (1.0 + gate)

