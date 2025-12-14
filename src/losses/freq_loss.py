import torch
import torch.nn as nn

class FFTAmplitudeLoss(nn.Module):
    """L1 loss between FFT amplitude spectra."""
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        # pred/target: (B, C, H, W)
        pred_fft = torch.fft.rfft2(pred, norm="ortho")
        tgt_fft = torch.fft.rfft2(target, norm="ortho")
        pred_amp = torch.abs(pred_fft)
        tgt_amp = torch.abs(tgt_fft)
        return torch.mean(torch.abs(pred_amp - tgt_amp))
