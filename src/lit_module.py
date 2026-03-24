import pytorch_lightning as pl
import torch

from .losses.charbonnier import CharbonnierLoss
from .losses.ssim import ssim as ssim_fn
from .losses.freq_loss import FFTAmplitudeLoss
from .metrics.psnr import psnr


class LitDerain(pl.LightningModule):
    def __init__(
        self,
        model,
        lr=2e-4,
        weight_decay=1e-5,
        loss_w=None,
        t_max=60,
        debug_first_n_batches=2,
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_w = loss_w or {"w_l1": 1.0, "w_ssim": 0.0, "w_fft": 0.0}
        self.t_max = int(t_max)
        self.range_mode = self.loss_w.get("range_mode", "01")
        self.debug_first_n_batches = int(debug_first_n_batches)

        self.l1 = CharbonnierLoss()
        self.fft_loss = FFTAmplitudeLoss()

        self.save_hyperparameters(ignore=["model"])

    def to_01(self, x):
        if self.range_mode == "m11":
            return (x.clamp(-1, 1) + 1.0) / 2.0
        return x.clamp(0.0, 1.0)

    def forward(self, x):
        return self.model(x)

    def _tensor_stats_str(self, x, name="tensor"):
        if not isinstance(x, torch.Tensor):
            return f"{name}: type={type(x)}"

        with torch.no_grad():
            finite_mask = torch.isfinite(x)
            finite_ratio = finite_mask.float().mean().item() if x.numel() > 0 else 1.0

            if finite_mask.any():
                xf = x[finite_mask]
                min_v = xf.min().item()
                max_v = xf.max().item()
                mean_v = xf.mean().item()
                std_v = xf.std().item() if xf.numel() > 1 else 0.0
            else:
                min_v = float("nan")
                max_v = float("nan")
                mean_v = float("nan")
                std_v = float("nan")

        return (
            f"{name}: shape={tuple(x.shape)}, dtype={x.dtype}, device={x.device}, "
            f"finite_ratio={finite_ratio:.6f}, min={min_v:.6f}, max={max_v:.6f}, "
            f"mean={mean_v:.6f}, std={std_v:.6f}"
        )

    def _debug_batch(self, batch, batch_idx, stage):
        if batch_idx >= self.debug_first_n_batches:
            return

        print(f"\n[DEBUG][{stage}] batch_idx={batch_idx}, batch_type={type(batch)}")

        if isinstance(batch, dict):
            print(f"[DEBUG][{stage}] batch keys = {list(batch.keys())}")
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    print("[DEBUG][" + stage + "] " + self._tensor_stats_str(v, name=k))
                else:
                    print(f"[DEBUG][{stage}] {k}: type={type(v)}")
        elif isinstance(batch, (list, tuple)):
            print(f"[DEBUG][{stage}] batch len = {len(batch)}")
            for i, v in enumerate(batch):
                if isinstance(v, torch.Tensor):
                    print("[DEBUG][" + stage + "] " + self._tensor_stats_str(v, name=f'item[{i}]'))
                else:
                    print(f"[DEBUG][{stage}] item[{i}]: type={type(v)}")
        else:
            print(f"[DEBUG][{stage}] unsupported batch preview type: {type(batch)}")

    def _extract_xy(self, batch):
        # Case 1: tuple/list
        if isinstance(batch, (list, tuple)):
            if len(batch) < 2:
                raise ValueError(f"Unsupported batch list/tuple length: {len(batch)}")
            return batch[0], batch[1]

        # Case 2: dict with common key pairs
        if isinstance(batch, dict):
            preferred_pairs = [
                ("rain", "gt"),
                ("rain", "clean"),
                ("rainy", "clean"),
                ("input", "target"),
                ("image", "target"),
                ("image", "gt"),
                ("x", "y"),
                ("noisy", "clean"),
                ("lr", "hr"),
            ]

            for kx, ky in preferred_pairs:
                if kx in batch and ky in batch:
                    return batch[kx], batch[ky]

            tensor_keys = [k for k, v in batch.items() if isinstance(v, torch.Tensor)]
            if len(tensor_keys) == 2:
                return batch[tensor_keys[0]], batch[tensor_keys[1]]

            raise ValueError(
                f"Unsupported dict batch format. keys={list(batch.keys())}, "
                f"tensor_keys={tensor_keys}"
            )

        raise ValueError(f"Unsupported batch format: {type(batch)}")

    def _check_tensor_finite(self, x, name):
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"{name} is not a torch.Tensor, got {type(x)}")

        if not torch.isfinite(x).all():
            msg = self._tensor_stats_str(x, name=name)
            raise RuntimeError(f"NaN/Inf detected in {name}\n{msg}")

    def _loss_terms(self, pred01, gt01):
        l1_term = self.l1(pred01, gt01)
        ssim_term = 1.0 - ssim_fn(pred01, gt01)
        fft_term = self.fft_loss(pred01, gt01)

        w1 = float(self.loss_w.get("w_l1", 1.0))
        w2 = float(self.loss_w.get("w_ssim", 0.0))
        w3 = float(self.loss_w.get("w_fft", 0.0))

        total = w1 * l1_term + w2 * ssim_term + w3 * fft_term
        return total, l1_term, ssim_term, fft_term

    def _shared_step(self, batch, batch_idx, stage="train"):
        self._debug_batch(batch, batch_idx, stage)

        rain, gt = self._extract_xy(batch)

        self._check_tensor_finite(rain, f"{stage}/rain_input")
        self._check_tensor_finite(gt, f"{stage}/gt_input")

        pred = self(rain)
        self._check_tensor_finite(pred, f"{stage}/pred_raw")

        pred01 = self.to_01(pred)
        gt01 = self.to_01(gt)

        self._check_tensor_finite(pred01, f"{stage}/pred01")
        self._check_tensor_finite(gt01, f"{stage}/gt01")

        loss, l1_term, ssim_term, fft_term = self._loss_terms(pred01, gt01)

        for name, t in [
            (f"{stage}/loss_total", loss),
            (f"{stage}/loss_l1", l1_term),
            (f"{stage}/loss_ssim", ssim_term),
            (f"{stage}/loss_fft", fft_term),
        ]:
            self._check_tensor_finite(t, name)

        if batch_idx < self.debug_first_n_batches:
            print("[DEBUG][" + stage + "] " + self._tensor_stats_str(rain, name="rain"))
            print("[DEBUG][" + stage + "] " + self._tensor_stats_str(gt, name="gt"))
            print("[DEBUG][" + stage + "] " + self._tensor_stats_str(pred, name="pred"))
            print(
                f"[DEBUG][{stage}] loss_total={loss.item():.6f}, "
                f"l1={l1_term.item():.6f}, ssim_loss={ssim_term.item():.6f}, "
                f"fft={fft_term.item():.6f}"
            )

        return rain, gt, pred01, gt01, loss, l1_term, ssim_term, fft_term

    def training_step(self, batch, batch_idx):
        rain, gt, pred01, gt01, loss, l1_term, ssim_term, fft_term = self._shared_step(
            batch, batch_idx, stage="train"
        )

        self.log(
            "train/loss_total",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=rain.size(0),
        )
        self.log(
            "train/loss_l1",
            l1_term,
            prog_bar=False,
            on_step=True,
            on_epoch=True,
            batch_size=rain.size(0),
        )
        self.log(
            "train/loss_ssim",
            ssim_term,
            prog_bar=False,
            on_step=True,
            on_epoch=True,
            batch_size=rain.size(0),
        )
        self.log(
            "train/loss_fft",
            fft_term,
            prog_bar=False,
            on_step=True,
            on_epoch=True,
            batch_size=rain.size(0),
        )

        return loss

    def validation_step(self, batch, batch_idx):
        rain, gt, pred01, gt01, loss, l1_term, ssim_term, fft_term = self._shared_step(
            batch, batch_idx, stage="val"
        )

        p = psnr(pred01, gt01)
        ssim_score = 1.0 - ssim_term

        self._check_tensor_finite(p, "val/psnr")
        self._check_tensor_finite(ssim_score, "val/ssim")

        self.log(
            "val/loss_total",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=rain.size(0),
        )
        self.log(
            "val/loss_l1",
            l1_term,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            batch_size=rain.size(0),
        )
        self.log(
            "val/loss_ssim",
            ssim_term,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            batch_size=rain.size(0),
        )
        self.log(
            "val/loss_fft",
            fft_term,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            batch_size=rain.size(0),
        )
        self.log(
            "val/psnr",
            p,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=rain.size(0),
        )
        self.log(
            "val/ssim",
            ssim_score,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=rain.size(0),
        )

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=max(1, self.t_max),
        )
        return {"optimizer": opt, "lr_scheduler": sch}