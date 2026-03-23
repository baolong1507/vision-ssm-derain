import torch
import torch.nn as nn
import torch.nn.functional as F


def _flatten_lr(x: torch.Tensor) -> torch.Tensor:
    # [B, C, H, W] -> [B, H*W, C]
    b, c, h, w = x.shape
    return x.permute(0, 2, 3, 1).reshape(b, h * w, c)


def _flatten_rl(x: torch.Tensor) -> torch.Tensor:
    return _flatten_lr(torch.flip(x, dims=[3]))


def _flatten_tb(x: torch.Tensor) -> torch.Tensor:
    # transpose H/W then flatten like lr
    xt = x.permute(0, 1, 3, 2).contiguous()  # [B,C,W,H]
    b, c, w, h = xt.shape
    return xt.permute(0, 2, 3, 1).reshape(b, w * h, c)


def _flatten_bt(x: torch.Tensor) -> torch.Tensor:
    xt = torch.flip(x.permute(0, 1, 3, 2).contiguous(), dims=[3])
    b, c, w, h = xt.shape
    return xt.permute(0, 2, 3, 1).reshape(b, w * h, c)


def _restore_from_seq(seq: torch.Tensor, h: int, w: int, direction: str) -> torch.Tensor:
    # seq: [B, N, C]
    b, n, c = seq.shape
    if direction == "lr":
        out = seq.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()
    elif direction == "rl":
        out = seq.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        out = torch.flip(out, dims=[3])
    elif direction == "tb":
        out = seq.reshape(b, w, h, c).permute(0, 3, 1, 2).contiguous()  # [B,C,W,H]
        out = out.permute(0, 1, 3, 2).contiguous()  # [B,C,H,W]
    elif direction == "bt":
        out = seq.reshape(b, w, h, c).permute(0, 3, 1, 2).contiguous()
        out = torch.flip(out, dims=[3])
        out = out.permute(0, 1, 3, 2).contiguous()
    else:
        raise ValueError(f"Unknown direction: {direction}")
    return out


class SimpleScan1D(nn.Module):
    """
    Lightweight recurrent scan:
      h_t = a_t * h_{t-1} + b_t * x_t
      y_t = proj([h_t, x_t])
    """
    def __init__(self, ch: int):
        super().__init__()
        self.to_a = nn.Linear(ch, ch)
        self.to_b = nn.Linear(ch, ch)
        self.out = nn.Linear(ch * 2, ch)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        # seq: [B, N, C]
        b, n, c = seq.shape
        state = torch.zeros((b, c), device=seq.device, dtype=seq.dtype)
        a = torch.sigmoid(self.to_a(seq))
        bcoef = torch.sigmoid(self.to_b(seq))

        ys = []
        for t in range(n):
            xt = seq[:, t, :]
            state = a[:, t, :] * state + bcoef[:, t, :] * xt
            yt = self.out(torch.cat([state, xt], dim=-1))
            ys.append(yt)

        return torch.stack(ys, dim=1)


class SymmetricScan2D(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.norm = nn.LayerNorm(ch)
        self.in_proj = nn.Linear(ch, ch)
        self.scan = SimpleScan1D(ch)
        self.out_proj = nn.Linear(ch, ch)
        self.mix = nn.Conv2d(ch, ch, 3, padding=1, groups=ch)

    def _run_one(self, seq: torch.Tensor) -> torch.Tensor:
        seq = self.norm(seq)
        seq = self.in_proj(seq)
        seq = self.scan(seq)
        seq = self.out_proj(seq)
        return seq

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape

        seq_lr = _flatten_lr(x)
        seq_rl = _flatten_rl(x)
        seq_tb = _flatten_tb(x)
        seq_bt = _flatten_bt(x)

        y_lr = _restore_from_seq(self._run_one(seq_lr), h, w, "lr")
        y_rl = _restore_from_seq(self._run_one(seq_rl), h, w, "rl")
        y_tb = _restore_from_seq(self._run_one(seq_tb), h, w, "tb")
        y_bt = _restore_from_seq(self._run_one(seq_bt), h, w, "bt")

        y = (y_lr + y_rl + y_tb + y_bt) / 4.0
        y = y + self.mix(y)
        return y


class SSM2DBlock(nn.Module):
    def __init__(self, ch, mode="convscan", kernel_size=31):
        super().__init__()
        self.mode = mode
        self.norm = nn.GroupNorm(8, ch)

        if mode == "convscan":
            self.dw = nn.Conv2d(ch, ch, kernel_size, padding=kernel_size // 2, groups=ch)
            self.pw = nn.Conv2d(ch, ch, 1)

        elif mode == "recurrent":
            self.gate = nn.Conv2d(ch, ch, 1)
            self.update = nn.Conv2d(ch, ch, 1)

        elif mode == "symscan":
            self.scan2d = SymmetricScan2D(ch)
            self.out_proj = nn.Conv2d(ch, ch, 1)

        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def forward(self, x):
        h = self.norm(x)

        if self.mode == "convscan":
            h = self.pw(F.relu(self.dw(h), inplace=True))
            return x + h

        if self.mode == "recurrent":
            b, c, hh, ww = h.shape
            gate = torch.sigmoid(self.gate(h))
            u = self.update(h)

            s = torch.zeros((b, c, hh, 1), device=h.device, dtype=h.dtype)
            out_w = []
            for t in range(ww):
                s = gate[:, :, :, t:t+1] * s + (1 - gate[:, :, :, t:t+1]) * u[:, :, :, t:t+1]
                out_w.append(s)
            hw = torch.cat(out_w, dim=3)

            s2 = torch.zeros((b, c, 1, ww), device=h.device, dtype=h.dtype)
            out_h = []
            for t in range(hh):
                s2 = gate[:, :, t:t+1, :] * s2 + (1 - gate[:, :, t:t+1, :]) * hw[:, :, t:t+1, :]
                out_h.append(s2)
            hh_out = torch.cat(out_h, dim=2)
            return x + hh_out

        if self.mode == "symscan":
            h = self.out_proj(self.scan2d(h))
            return x + h

        raise ValueError(f"Unsupported mode in forward: {self.mode}")