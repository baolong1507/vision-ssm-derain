# src\losses\freq_loss.py
import torch
import torch.nn as nn


class FFTAmplitudeLoss(nn.Module):
    def __init__(self, eps=1e-8, use_log_amp=True):
        super().__init__()
        self.eps = float(eps)
        self.use_log_amp = bool(use_log_amp)

    def forward(self, pred, gt):
        with torch.amp.autocast(device_type=pred.device.type, enabled=False):
            pred32 = pred.float()
            gt32 = gt.float()

            Fp = torch.fft.rfft2(pred32, norm="ortho")
            Fg = torch.fft.rfft2(gt32, norm="ortho")

            amp_p = torch.abs(Fp)
            amp_g = torch.abs(Fg)

            amp_p = torch.nan_to_num(amp_p, nan=0.0, posinf=1e6, neginf=0.0)
            amp_g = torch.nan_to_num(amp_g, nan=0.0, posinf=1e6, neginf=0.0)

            if self.use_log_amp:
                amp_p = torch.log1p(torch.clamp(amp_p, min=0.0))
                amp_g = torch.log1p(torch.clamp(amp_g, min=0.0))

            diff = torch.abs(amp_p - amp_g)
            diff = torch.nan_to_num(diff, nan=0.0, posinf=1e6, neginf=1e6)

            loss = diff.mean()
            loss = torch.nan_to_num(loss, nan=0.0, posinf=1e6, neginf=1e6)

        return loss
