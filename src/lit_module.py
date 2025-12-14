import pytorch_lightning as pl
import torch
from .losses.charbonnier import CharbonnierLoss
from .losses.ssim import ssim as ssim_fn
from .losses.freq_loss import FFTAmplitudeLoss
from .metrics.psnr import psnr

class LitDerain(pl.LightningModule):
    def __init__(self, model, lr=2e-4, weight_decay=1e-5, loss_w=None):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_w = loss_w or {"w_l1": 1.0, "w_ssim": 0.0, "w_fft": 0.0}

        self.l1 = CharbonnierLoss()
        self.fft_loss = FFTAmplitudeLoss()

    def forward(self, x):
        return self.model(x)

    def _loss(self, pred, gt):
        w1 = float(self.loss_w.get("w_l1", 1.0))
        w2 = float(self.loss_w.get("w_ssim", 0.0))
        w3 = float(self.loss_w.get("w_fft", 0.0))

        loss = 0.0
        if w1 > 0:
            loss = loss + w1 * self.l1(pred, gt)
        if w2 > 0:
            loss = loss + w2 * (1.0 - ssim_fn(pred, gt))
        if w3 > 0:
            loss = loss + w3 * self.fft_loss(pred, gt)
        return loss

    def training_step(self, batch, batch_idx):
        rain, gt = batch["rain"], batch["gt"]
        pred = self(rain)
        loss = self._loss(pred, gt)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        rain, gt = batch["rain"], batch["gt"]
        pred = self(rain)
        loss = self._loss(pred, gt)
        p = psnr(pred, gt)
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/psnr", p, prog_bar=True)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, self.trainer.max_epochs))
        return {"optimizer": opt, "lr_scheduler": sch}
