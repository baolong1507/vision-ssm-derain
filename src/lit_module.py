import pytorch_lightning as pl
import torch
from .losses.charbonnier import CharbonnierLoss
from .losses.ssim import ssim as ssim_fn
from .losses.freq_loss import FFTAmplitudeLoss
from .metrics.psnr import psnr

class LitDerain(pl.LightningModule):
    def __init__(self, model, lr=2e-4, weight_decay=1e-5, loss_w=None, t_max=60):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_w = loss_w or {"w_l1": 1.0, "w_ssim": 0.0, "w_fft": 0.0}
        self.t_max = int(t_max)

        self.range_mode = self.loss_w.get("range_mode", "01")  # "01" or "m11"

        self.l1 = CharbonnierLoss()
        self.fft_loss = FFTAmplitudeLoss()

    def to_01(self, x):
        if self.range_mode == "m11":
            return (x.clamp(-1, 1) + 1) / 2
        return x.clamp(0, 1)

    def forward(self, x):
        return self.model(x)

    def _loss(self, pred01, gt01):
        w1 = float(self.loss_w.get("w_l1", 1.0))
        w2 = float(self.loss_w.get("w_ssim", 0.0))
        w3 = float(self.loss_w.get("w_fft", 0.0))

        loss = 0.0
        if w1 > 0:
            loss = loss + w1 * self.l1(pred01, gt01)
        if w2 > 0:
            loss = loss + w2 * (1.0 - ssim_fn(pred01, gt01))
        if w3 > 0:
            loss = loss + w3 * self.fft_loss(pred01, gt01)
        return loss

    def training_step(self, batch, batch_idx):
        rain, gt = batch["rain"], batch["gt"]
        pred = self(rain)

        if not torch.isfinite(pred).all():
            raise RuntimeError("NaN/Inf in pred (model forward)")

        pred01 = self.to_01(pred)
        gt01   = self.to_01(gt)

        loss = self._loss(pred01, gt01)
        if not torch.isfinite(loss):
            raise RuntimeError("NaN/Inf in loss")

        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=rain.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        rain, gt = batch["rain"], batch["gt"]
        pred = self(rain)

        if not torch.isfinite(pred).all():
            raise RuntimeError("NaN/Inf in pred (val forward)")

        pred01 = self.to_01(pred)
        gt01   = self.to_01(gt)

        loss = self._loss(pred01, gt01)
        if not torch.isfinite(loss):
            raise RuntimeError("NaN/Inf in val loss")

        p = psnr(pred01, gt01)
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=rain.size(0))
        self.log("val/psnr", p, prog_bar=True, on_step=False, on_epoch=True, batch_size=rain.size(0))

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, self.t_max))
        return {"optimizer": opt, "lr_scheduler": sch}