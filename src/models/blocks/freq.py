# src\models\blocks\freq.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class FreqEnhance(nn.Module):
    """
    Frequency enhancement block with safer numerical handling.

    Strategy:
    - Run FFT in float32 to avoid ComplexHalf instability under AMP/fp16
    - Compute amplitude map
    - Resize to spatial size
    - Generate gate and inject back to input
    """

    def __init__(self, in_ch, freq_ch=16, eps=1e-6, debug=False):
        super().__init__()
        self.eps = float(eps)
        self.debug = bool(debug)

        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, freq_ch, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(freq_ch, in_ch, 1),
            nn.Sigmoid(),
        )

    def _check_finite(self, x, name):
        if not torch.isfinite(x).all():
            finite_mask = torch.isfinite(x)
            if finite_mask.any():
                xf = x[finite_mask]
                min_v = xf.min().item()
                max_v = xf.max().item()
                mean_v = xf.mean().item()
            else:
                min_v = float("nan")
                max_v = float("nan")
                mean_v = float("nan")

            raise RuntimeError(
                f"[FreqEnhance] NaN/Inf detected in {name}: "
                f"shape={tuple(x.shape)}, dtype={x.dtype}, "
                f"min={min_v:.6f}, max={max_v:.6f}, mean={mean_v:.6f}"
            )

    def forward(self, x):
        # Keep residual path in original dtype
        x_in = x
        orig_dtype = x.dtype

        # FFT path in float32 for stability
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            x32 = x.float()
            self._check_finite(x32, "x32_before_fft")

            fft = torch.fft.rfft2(x32, norm="ortho")
            amp = torch.abs(fft)

            # Prevent NaN/Inf from propagating
            amp = torch.nan_to_num(amp, nan=0.0, posinf=1e6, neginf=0.0)
            self._check_finite(amp, "amp_after_abs")

            # Resize from (B,C,H,W//2+1) -> (B,C,H,W)
            amp = F.interpolate(
                amp,
                size=x.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

            # Optional compression to reduce dynamic range spikes
            amp = torch.log1p(torch.clamp(amp, min=0.0))
            amp = torch.nan_to_num(amp, nan=0.0, posinf=20.0, neginf=0.0)
            self._check_finite(amp, "amp_after_interp_log")

        # Conv branch can run in normal dtype/autocast context
        gate = self.proj(amp.to(orig_dtype))
        gate = torch.nan_to_num(gate, nan=0.0, posinf=1.0, neginf=0.0)

        # Sigmoid should keep it in [0,1], but clamp for extra safety
        gate = torch.clamp(gate, 0.0, 1.0)

        out = x_in * (1.0 + gate)

        if self.debug:
            if not torch.isfinite(out).all():
                self._check_finite(out, "out")

        return out

