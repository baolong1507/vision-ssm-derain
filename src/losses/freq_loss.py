import torch
import torch.nn as nn

class FFTAmplitudeLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred, gt):
        # pred/gt: (B,C,H,W) có thể đang float16 do AMP
        with torch.amp.autocast(enabled=False):
            pred32 = pred.float()
            gt32   = gt.float()

            Fp = torch.fft.rfft2(pred32, norm="ortho")
            Fg = torch.fft.rfft2(gt32, norm="ortho")

            amp_p = torch.abs(Fp)
            amp_g = torch.abs(Fg)

            loss = torch.mean(torch.abs(amp_p - amp_g))

            # (optional) chặn NaN lan ra ngoài
            loss = torch.nan_to_num(loss, nan=0.0, posinf=1e6, neginf=1e6)

        return loss
