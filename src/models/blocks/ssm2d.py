import torch
import torch.nn as nn
import torch.nn.functional as F

class SSM2DBlock(nn.Module):
    def __init__(self, ch, mode="convscan", kernel_size=31):
        super().__init__()
        self.mode = mode
        self.norm = nn.GroupNorm(8, ch)

        if mode == "convscan":
            # depthwise conv to mix long-range context
            self.dw = nn.Conv2d(ch, ch, kernel_size, padding=kernel_size//2, groups=ch)
            self.pw = nn.Conv2d(ch, ch, 1)
        else:
            # recurrent-like scan: row then col
            self.gate = nn.Conv2d(ch, ch, 1)
            self.update = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        h = self.norm(x)
        if self.mode == "convscan":
            h = self.pw(F.relu(self.dw(h), inplace=True))
            return x + h
        else:
            # recurrent scan (simple)
            B, C, H, W = h.shape
            gate = torch.sigmoid(self.gate(h))
            u = self.update(h)

            # scan width
            s = torch.zeros((B, C, H, 1), device=h.device, dtype=h.dtype)
            out_w = []
            for t in range(W):
                s = gate[:, :, :, t:t+1] * s + (1 - gate[:, :, :, t:t+1]) * u[:, :, :, t:t+1]
                out_w.append(s)
            hw = torch.cat(out_w, dim=3)

            # scan height
            s2 = torch.zeros((B, C, 1, W), device=h.device, dtype=h.dtype)
            out_h = []
            for t in range(H):
                s2 = gate[:, :, t:t+1, :] * s2 + (1 - gate[:, :, t:t+1, :]) * hw[:, :, t:t+1, :]
                out_h.append(s2)
            hh = torch.cat(out_h, dim=2)

            return x + hh
