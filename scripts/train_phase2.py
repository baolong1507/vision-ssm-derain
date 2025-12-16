import argparse
from pathlib import Path

import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.utilities import rank_zero_only

from src.utils.seed import seed_everything
from src.data.datamodule import DerainDataModule
from src.lit_module import LitDerain
from src.models.fess_unet import FESSUNet


def ensure_dir(p: str) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def warm_start_from_ckpt(model: torch.nn.Module, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    def strip_prefix(k: str):
        for p in ("model.", "net.", "module."):
            if k.startswith(p):
                return k[len(p):]
        return k

    sd = {strip_prefix(k): v for k, v in sd.items()}

    cur = model.state_dict()
    matched = {}
    skip_name = 0
    skip_shape = 0

    for k, v in sd.items():
        if k not in cur:
            skip_name += 1
            continue
        if tuple(v.shape) != tuple(cur[k].shape):
            skip_shape += 1
            continue
        matched[k] = v

    model.load_state_dict(matched, strict=False)
    print(f"[WarmStart] Loaded {len(matched)}/{len(cur)} tensors from: {ckpt_path}")
    print(f"[WarmStart] Skipped (name not found): {skip_name}")
    print(f"[WarmStart] Skipped (shape mismatch): {skip_shape}")


def optional_zero_init_freq_proj(model: torch.nn.Module):
    """
    Zero-init proj giúp gate bắt đầu ~0 => ổn định gating.
    """
    import torch.nn as nn

    n = 0
    for m in model.modules():
        if hasattr(m, "proj"):
            proj = getattr(m, "proj")
            if isinstance(proj, nn.Conv2d):
                nn.init.zeros_(proj.weight)
                if proj.bias is not None:
                    nn.init.zeros_(proj.bias)
                n += 1
            elif isinstance(proj, nn.Sequential):
                # zero init conv cuối nếu có
                conv_last = None
                for layer in proj.modules():
                    if isinstance(layer, nn.Conv2d):
                        conv_last = layer
                if conv_last is not None:
                    nn.init.zeros_(conv_last.weight)
                    if conv_last.bias is not None:
                        nn.init.zeros_(conv_last.bias)
                    n += 1
    print(f"[Init] Zero-initialized {n} proj conv(s) to stabilize gating.")


def freeze_gates(model: torch.nn.Module, freeze: bool = True):
    """
    Freeze/unfreeze các tham số thuộc proj (gate) trong freq blocks.
    Dùng để warm-up 0-1 epoch đầu cho ổn định.
    """
    n = 0
    for m in model.modules():
        if hasattr(m, "proj"):
            for p in getattr(m, "proj").parameters():
                p.requires_grad = (not freeze)
                n += p.numel()
    print(f"[GateFreeze] {'FROZEN' if freeze else 'UNFROZEN'} proj params ({n} elements).")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--init_from", default="")
    ap.add_argument("--resume", default="")  # last | best | /path
    ap.add_argument("--run_name", default="")
    ap.add_argument("--precision", default="")  # 16-mixed | bf16-mixed | 32
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--zero_init_gate", action="store_true")
    ap.add_argument("--freeze_gate_epochs", type=int, default=1,
                    help="freeze gate/proj for first N epochs to avoid early divergence (default=1)")
    ap.add_argument("--terminate_on_nan", action="store_true",
                    help="trainer will stop if loss becomes NaN/Inf")
    ap.add_argument("--detect_anomaly", action="store_true",
                    help="very slow but good for debugging NaNs")

    ap.add_argument("--no_sanity", action="store_true",
                    help="disable sanity val steps to avoid confusing val/psnr early")
    return ap.parse_args()


def resolve_ckpt_path(ckpt_dir: Path, resume: str) -> str:
    if not resume:
        return ""
    if resume == "last":
        p = ckpt_dir / "last.ckpt"
        if not p.exists():
            raise FileNotFoundError(f"Resume asked last but not found: {p}")
        return str(p)
    if resume == "best":
        # pick highest val/psnr from filename "epochXXX-psnrYY.YY.ckpt"
        cand = list(ckpt_dir.glob("epoch*-psnr*.ckpt"))
        if not cand:
            raise FileNotFoundError(f"Resume asked best but no epoch*-psnr*.ckpt in {ckpt_dir}")

        def parse_psnr(p: Path):
            s = p.stem  # epoch000-psnr28.53
            try:
                return float(s.split("psnr")[-1])
            except Exception:
                return -1e9

        cand = sorted(cand, key=parse_psnr)
        return str(cand[-1])

    p = Path(resume)
    if not p.exists():
        raise FileNotFoundError(f"Resume ckpt not found: {p}")
    return str(p)


class GateWarmupCallback(pl.Callback):
    """
    Freeze gate/proj trong vài epoch đầu, rồi unfreeze.
    """
    def __init__(self, n_epochs_freeze: int):
        super().__init__()
        self.n = max(0, int(n_epochs_freeze))

    def on_fit_start(self, trainer, pl_module):
        if self.n > 0:
            freeze_gates(pl_module.model, freeze=True)

    def on_train_epoch_start(self, trainer, pl_module):
        if self.n > 0 and trainer.current_epoch == self.n:
            freeze_gates(pl_module.model, freeze=False)


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.cfg)
    data_cfg = OmegaConf.load(args.data)

    seed_everything(int(args.seed))

    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    ckpt_dir = Path(cfg.output.ckpt_dir)
    res_dir = Path(cfg.output.result_dir)

    if args.run_name:
        ckpt_dir = ckpt_dir / args.run_name
        res_dir = res_dir / args.run_name

    ckpt_dir = ensure_dir(str(ckpt_dir))
    res_dir = ensure_dir(str(res_dir))

    print("[Paths] ckpt_dir:", ckpt_dir)
    print("[Paths] result_dir:", res_dir)

    # ---- build model ----
    model = FESSUNet(base_ch=int(cfg.model.base_ch), freq_ch=int(cfg.model.freq_ch))

    if args.zero_init_gate:
        optional_zero_init_freq_proj(model)

    if args.init_from:
        warm_start_from_ckpt(model, args.init_from)

    lit = LitDerain(
        model=model,
        lr=float(cfg.train.lr),
        weight_decay=float(cfg.train.weight_decay),
        loss_w=dict(cfg.loss),
    )

    # ---- data ----
    dm = DerainDataModule(data_cfg=data_cfg, train_cfg=cfg, cfg=None)
    train_tfms, val_tfms = dm.build_transforms()
    dm.setup(train_tfms, val_tfms)
    train_loader, val_loader = dm.train_dataloader(), dm.val_dataloader()

    # ---- callbacks ----
    ckpt_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="epoch{epoch:03d}-psnr{val/psnr:.2f}",
        monitor="val/psnr",
        mode="max",
        save_top_k=2,
        save_last=True,
        every_n_epochs=1,
    )

    precision = args.precision.strip() if args.precision else str(cfg.train.get("precision", "bf16-mixed"))
    if precision not in ("16-mixed", "bf16-mixed", "32"):
        precision = "bf16-mixed"

    callbacks = [ckpt_cb, LearningRateMonitor(logging_interval="epoch")]

    # Gate warmup (rất hiệu quả chống diverge/NaN)
    if args.freeze_gate_epochs > 0:
        callbacks.append(GateWarmupCallback(args.freeze_gate_epochs))

    trainer = pl.Trainer(
        max_epochs=int(cfg.train.max_epochs),
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision=precision,
        callbacks=callbacks,
        enable_checkpointing=True,
        default_root_dir=str(ckpt_dir.parent),
        log_every_n_steps=50,
        gradient_clip_val=float(cfg.train.get("grad_clip", 0.5)),
        gradient_clip_algorithm="norm",
        num_sanity_val_steps=0 if args.no_sanity else 2,
        terminate_on_nan=args.terminate_on_nan,
        detect_anomaly=args.detect_anomaly,
    )

    ckpt_path = resolve_ckpt_path(ckpt_dir, args.resume)
    if ckpt_path:
        print("[Resume] from:", ckpt_path)

    trainer.fit(lit, train_loader, val_loader, ckpt_path=ckpt_path if ckpt_path else None)


if __name__ == "__main__":
    main()
