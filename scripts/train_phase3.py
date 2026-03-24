import argparse
from pathlib import Path

from omegaconf import OmegaConf

from src.utils.seed import seed_everything
from src.data.datamodule import DerainDataModule
from src.data.transforms_albu import build_transforms
from src.models.fessm_net import FESSMNet
from src.lit_module import LitDerain
from src.utils.io import ensure_dir

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

import torch

torch.set_float32_matmul_precision("high")


def _cfg_get(cfg_node, key, default=None):
    """
    Safe getter cho OmegaConf DictConfig hoặc dict thường.
    Không crash khi thiếu key.
    """
    try:
        if cfg_node is None:
            return default
        if isinstance(cfg_node, dict):
            return cfg_node.get(key, default)
        return cfg_node.get(key, default)
    except Exception:
        try:
            return getattr(cfg_node, key)
        except Exception:
            return default


def main(cfg_path, data_cfg_path):
    cfg = OmegaConf.load(cfg_path)
    data_cfg = OmegaConf.load(data_cfg_path)

    seed_everything(int(_cfg_get(cfg, "seed", 42)))

    img_size = int(_cfg_get(data_cfg, "img_size", 256))
    crop_size = int(_cfg_get(data_cfg, "crop_size", img_size))

    train_tfms = build_transforms(img_size, crop_size, True)
    val_tfms = build_transforms(img_size, crop_size, False)

    dm = DerainDataModule(
        data_cfg=data_cfg,
        train_cfg={
            "batch_size": int(_cfg_get(cfg.train, "batch_size", 1)),
            "auto_split_val": True,
            "val_ratio": 0.1,
            "split_seed": int(_cfg_get(cfg, "seed", 42)),
        },
        cfg=cfg,
    )
    dm.setup(train_tfms, val_tfms)
    train_loader = dm.train_loader()
    val_loader = dm.val_loader()

    model = FESSMNet(
        base_ch=int(_cfg_get(cfg.model, "base_ch", 48)),
        freq_ch=int(_cfg_get(cfg.model, "freq_ch", 16)),
        ssm_mode=str(_cfg_get(cfg.model, "ssm_mode", "convscan")),
        use_freq=bool(_cfg_get(cfg.model, "use_freq", False)),
        use_ssm_bottleneck=bool(_cfg_get(cfg.model, "use_ssm_bottleneck", True)),
        use_ssm_decoder=bool(_cfg_get(cfg.model, "use_ssm_decoder", True)),
    )

    lit = LitDerain(
        model=model,
        lr=float(_cfg_get(cfg.train, "lr", 2e-4)),
        weight_decay=float(_cfg_get(cfg.train, "weight_decay", 1e-5)),
        loss_w=dict(_cfg_get(cfg, "loss", {})),
        t_max=int(_cfg_get(cfg.train, "max_epochs", 1)),
    )

    ckpt_dir = ensure_dir(_cfg_get(cfg.output, "ckpt_dir", "outputs/ckpts/phase3"))

    ckpt = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="epoch{epoch:03d}-psnr{val/psnr:.2f}",
        monitor="val/psnr",
        mode="max",
        save_top_k=2,
        save_last=True,
    )

    limit_train_batches = _cfg_get(cfg.train, "limit_train_batches", 1.0)
    limit_val_batches = _cfg_get(cfg.train, "limit_val_batches", 1.0)
    num_sanity_val_steps = int(_cfg_get(cfg.train, "num_sanity_val_steps", 2))
    grad_clip = float(_cfg_get(cfg.train, "grad_clip", 0.0))

    trainer = pl.Trainer(
        max_epochs=int(_cfg_get(cfg.train, "max_epochs", 1)),
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision=str(_cfg_get(cfg.train, "precision", "16-mixed")),
        callbacks=[ckpt, LearningRateMonitor(logging_interval="epoch")],
        enable_checkpointing=True,
        default_root_dir=str(Path(_cfg_get(cfg.output, "ckpt_dir", "outputs/ckpts/phase3")).parents[0]),
        log_every_n_steps=int(_cfg_get(cfg.train, "log_every_n_steps", 10)),
        gradient_clip_val=grad_clip,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        num_sanity_val_steps=num_sanity_val_steps,
    )

    print("========== TRAIN CONFIG ==========")
    print(f"cfg_path             : {cfg_path}")
    print(f"data_cfg_path        : {data_cfg_path}")
    print(f"accelerator          : {'gpu' if torch.cuda.is_available() else 'cpu'}")
    print(f"max_epochs           : {int(_cfg_get(cfg.train, 'max_epochs', 1))}")
    print(f"batch_size           : {int(_cfg_get(cfg.train, 'batch_size', 1))}")
    print(f"limit_train_batches  : {limit_train_batches}")
    print(f"limit_val_batches    : {limit_val_batches}")
    print(f"num_sanity_val_steps : {num_sanity_val_steps}")
    print(f"precision            : {str(_cfg_get(cfg.train, 'precision', '16-mixed'))}")
    print(f"ssm_mode             : {str(_cfg_get(cfg.model, 'ssm_mode', 'convscan'))}")
    print(f"use_freq             : {bool(_cfg_get(cfg.model, 'use_freq', False))}")
    print("==================================")

    trainer.fit(lit, train_loader, val_loader)

    print("Saved ckpts to:", ckpt_dir)
    print(list(ckpt_dir.glob("*.ckpt"))[:5])

    manual = Path(ckpt_dir) / "manual_last.ckpt"
    trainer.save_checkpoint(str(manual))
    print("Manual checkpoint:", manual)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True, help="Path to training config yaml")
    ap.add_argument("--data", default="configs/data_config.yaml", help="Path to data config yaml")
    args = ap.parse_args()
    main(args.cfg, args.data)