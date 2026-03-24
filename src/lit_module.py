import torch
import torch.nn as nn
import pytorch_lightning as pl

from torchmetrics.functional.image import peak_signal_noise_ratio, structural_similarity_index_measure


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps ** 2))


class FFTAmplitudeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        pred_fft = torch.fft.rfft2(pred, norm="ortho")
        tgt_fft = torch.fft.rfft2(target, norm="ortho")
        pred_amp = torch.abs(pred_fft)
        tgt_amp = torch.abs(tgt_fft)
        return torch.mean(torch.abs(pred_amp - tgt_amp))


class LitDerain(pl.LightningModule):
    def __init__(self, model, lr=2e-4, weight_decay=1e-5, loss_w=None, t_max=50):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.t_max = t_max

        self.loss_w = loss_w or {}
        self.w_l1 = float(self.loss_w.get("w_l1", 1.0))
        self.w_ssim = float(self.loss_w.get("w_ssim", 0.2))
        self.w_fft = float(self.loss_w.get("w_fft", 0.0))

        self.loss_l1 = CharbonnierLoss()
        self.loss_fft = FFTAmplitudeLoss()

        self.save_hyperparameters(ignore=["model"])

    # =========================
    # Debug helpers
    # =========================
    def _safe_stats(self, t: torch.Tensor):
        t_det = t.detach()
        return {
            "shape": tuple(t_det.shape),
            "dtype": str(t_det.dtype),
            "device": str(t_det.device),
            "min": float(t_det.min().item()) if t_det.numel() > 0 else 0.0,
            "max": float(t_det.max().item()) if t_det.numel() > 0 else 0.0,
            "mean": float(t_det.mean().item()) if t_det.numel() > 0 else 0.0,
            "std": float(t_det.std().item()) if t_det.numel() > 1 else 0.0,
            "has_nan": bool(torch.isnan(t_det).any().item()),
            "has_inf": bool(torch.isinf(t_det).any().item()),
        }

    def _print_tensor_stats(self, name: str, t: torch.Tensor, stage: str, batch_idx: int):
        s = self._safe_stats(t)
        print(
            f"[{stage}] batch={batch_idx} {name}: "
            f"shape={s['shape']} dtype={s['dtype']} device={s['device']} "
            f"min={s['min']:.6f} max={s['max']:.6f} mean={s['mean']:.6f} std={s['std']:.6f} "
            f"nan={s['has_nan']} inf={s['has_inf']}"
        )

    def _assert_finite_tensor(self, name: str, t: torch.Tensor, stage: str, batch_idx: int):
        has_nan = torch.isnan(t).any()
        has_inf = torch.isinf(t).any()
        if has_nan or has_inf:
            self._print_tensor_stats(name, t, stage, batch_idx)
            raise RuntimeError(f"NaN/Inf detected in {name} during {stage} at batch_idx={batch_idx}")

    def _extract_xy(self, batch):
        """
        Hỗ trợ nhiều kiểu batch:
        - (x, y)
        - (x, y, meta)
        - {"input": x, "target": y}
        - {"rain": x, "clean": y}
        - {"x": x, "y": y}
        """
        if isinstance(batch, (list, tuple)):
            if len(batch) >= 2:
                return batch[0], batch[1]

        if isinstance(batch, dict):
            if "input" in batch and "target" in batch:
                return batch["input"], batch["target"]
            if "rain" in batch and "clean" in batch:
                return batch["rain"], batch["clean"]
            if "x" in batch and "y" in batch:
                return batch["x"], batch["y"]

        raise ValueError(f"Unsupported batch format: {type(batch)}")

    # =========================
    # Core logic
    # =========================
    def forward(self, x):
        return self.model(x)

    def _compute_losses_and_metrics(self, pred, y):
        l1 = self.loss_l1(pred, y)

        ssim_val = structural_similarity_index_measure(pred, y, data_range=1.0)
        ssim_loss = 1.0 - ssim_val

        if self.w_fft > 0:
            fft_loss = self.loss_fft(pred, y)
        else:
            fft_loss = torch.zeros((), device=pred.device, dtype=pred.dtype)

        total = self.w_l1 * l1 + self.w_ssim * ssim_loss + self.w_fft * fft_loss
        psnr = peak_signal_noise_ratio(pred, y, data_range=1.0)

        return {
            "loss_total": total,
            "loss_l1": l1,
            "loss_ssim": ssim_loss,
            "loss_fft": fft_loss,
            "psnr": psnr,
            "ssim": ssim_val,
        }

    def _shared_step(self, batch, batch_idx, stage="train"):
        x, y = self._extract_xy(batch)

        # Debug input
        self._assert_finite_tensor("x", x, stage, batch_idx)
        self._assert_finite_tensor("y", y, stage, batch_idx)

        # Optional: print first few batches only
        if batch_idx < 3:
            self._print_tensor_stats("x", x, stage, batch_idx)
            self._print_tensor_stats("y", y, stage, batch_idx)

        pred = self(x)

        # Debug pred
        self._assert_finite_tensor("pred", pred, stage, batch_idx)
        if batch_idx < 3:
            self._print_tensor_stats("pred", pred, stage, batch_idx)

        pred = pred.clamp(0.0, 1.0)
        y = y.clamp(0.0, 1.0)

        out = self._compute_losses_and_metrics(pred, y)

        # Debug losses
        for k in ["loss_total", "loss_l1", "loss_ssim", "loss_fft", "psnr", "ssim"]:
            v = out[k]
            if torch.isnan(v) or torch.isinf(v):
                print(f"[{stage}] batch={batch_idx} {k} is invalid: {v}")
                self._print_tensor_stats("x", x, stage, batch_idx)
                self._print_tensor_stats("y", y, stage, batch_idx)
                self._print_tensor_stats("pred", pred, stage, batch_idx)
                raise RuntimeError(f"NaN/Inf detected in {k} during {stage} at batch_idx={batch_idx}")

        return out

    # =========================
    # Lightning steps
    # =========================
    def training_step(self, batch, batch_idx):
        out = self._shared_step(batch, batch_idx, stage="train")

        self.log("train/loss_total_step", out["loss_total"], prog_bar=False, on_step=True, on_epoch=False, batch_size=batch[0].size(0) if isinstance(batch, (list, tuple)) else None)
        self.log("train/loss_total_epoch", out["loss_total"], prog_bar=True, on_step=False, on_epoch=True, batch_size=batch[0].size(0) if isinstance(batch, (list, tuple)) else None)
        self.log("train/psnr", out["psnr"], prog_bar=False, on_step=False, on_epoch=True, batch_size=batch[0].size(0) if isinstance(batch, (list, tuple)) else None)
        self.log("train/ssim", out["ssim"], prog_bar=False, on_step=False, on_epoch=True, batch_size=batch[0].size(0) if isinstance(batch, (list, tuple)) else None)

        return out["loss_total"]

    def validation_step(self, batch, batch_idx):
        out = self._shared_step(batch, batch_idx, stage="val")

        self.log("val/loss_total", out["loss_total"], prog_bar=True, on_step=False, on_epoch=True, batch_size=batch[0].size(0) if isinstance(batch, (list, tuple)) else None)
        self.log("val/psnr", out["psnr"], prog_bar=True, on_step=False, on_epoch=True, batch_size=batch[0].size(0) if isinstance(batch, (list, tuple)) else None)
        self.log("val/ssim", out["ssim"], prog_bar=True, on_step=False, on_epoch=True, batch_size=batch[0].size(0) if isinstance(batch, (list, tuple)) else None)

        return out["loss_total"]

    def test_step(self, batch, batch_idx):
        out = self._shared_step(batch, batch_idx, stage="test")

        self.log("test/loss_total", out["loss_total"], prog_bar=True, on_step=False, on_epoch=True, batch_size=batch[0].size(0) if isinstance(batch, (list, tuple)) else None)
        self.log("test/psnr", out["psnr"], prog_bar=True, on_step=False, on_epoch=True, batch_size=batch[0].size(0) if isinstance(batch, (list, tuple)) else None)
        self.log("test/ssim", out["ssim"], prog_bar=True, on_step=False, on_epoch=True, batch_size=batch[0].size(0) if isinstance(batch, (list, tuple)) else None)

        return out["loss_total"]

    # =========================
    # Optimizer
    # =========================
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.t_max,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }