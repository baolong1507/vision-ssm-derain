import argparse
from pathlib import Path
from omegaconf import OmegaConf
from src.utils.seed import seed_everything
from src.data.datamodule import DerainDataModule
from src.data.transforms_albu import build_transforms
from src.models.fess_unet import FESSUNet
from src.lit_module import LitDerain
from src.utils.io import ensure_dir
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import torch
torch.set_float32_matmul_precision("high")

def warm_start_from_ckpt(model: torch.nn.Module, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # Lightning ckpt thường có key 'state_dict'
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    # Nhiều repo lưu model dưới prefix 'model.'
    # Ví dụ: 'model.enc1.0.weight' -> 'enc1.0.weight'
    def strip_prefix(k: str):
        for p in ("model.", "net.", "module."):
            if k.startswith(p):
                return k[len(p):]
        return k

    sd_stripped = {strip_prefix(k): v for k, v in sd.items()}

    cur = model.state_dict()
    matched = {}
    skipped_name = 0
    skipped_shape = 0

    for k, v in sd_stripped.items():
        if k not in cur:
            skipped_name += 1
            continue
        if tuple(v.shape) != tuple(cur[k].shape):
            skipped_shape += 1
            continue
        matched[k] = v

    model.load_state_dict(matched, strict=False)

    print(f"[WarmStart] Loaded {len(matched)}/{len(cur)} tensors from: {ckpt_path}")
    print(f"[WarmStart] Skipped (name not found): {skipped_name}")
    print(f"[WarmStart] Skipped (shape mismatch): {skipped_shape}")

def main(cfg_path, data_cfg_path):
    cfg = OmegaConf.load(cfg_path)
    data_cfg = OmegaConf.load(data_cfg_path)

    seed_everything(int(cfg.seed))

    train_tfms = build_transforms(int(data_cfg.img_size), int(data_cfg.crop_size), True)
    val_tfms   = build_transforms(int(data_cfg.img_size), int(data_cfg.crop_size), False)

    dm = DerainDataModule(data_cfg, train_cfg={
        "batch_size": int(cfg.train.batch_size),
        "auto_split_val": True,       # 
        "val_ratio": 0.1,             # 
        "split_seed": int(cfg.seed),  # 
    })

    dm.setup(train_tfms, val_tfms)
    train_loader = dm.train_loader()
    val_loader   = dm.val_loader()

    model = FESSUNet(base_ch=int(cfg.model.base_ch), freq_ch=int(cfg.model.freq_ch))

    if args.init_from:
        warm_start_from_ckpt(model, args.init_from)

    lit = LitDerain(model=model, lr=float(cfg.train.lr),
                    weight_decay=float(cfg.train.weight_decay),
                    loss_w=dict(cfg.loss))

    ckpt_dir = ensure_dir(cfg.output.ckpt_dir)
    ckpt = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="epoch{epoch:03d}-psnr{val/psnr:.2f}",
        monitor="val/psnr",
        mode="max",
        save_top_k=2,
        save_last=True,
    )

    trainer = pl.Trainer(
        max_epochs=int(cfg.train.max_epochs),
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="16-mixed",
        callbacks=[ckpt, LearningRateMonitor(logging_interval="epoch")],
        enable_checkpointing=True,        # <<< quan trọng
        default_root_dir=str(Path(cfg.output.ckpt_dir).parents[0]),
        log_every_n_steps=50,
        gradient_clip_val=1.0,  
        gradient_clip_algorithm="norm",
    )

    trainer.fit(lit, train_loader, val_loader)
    print("Saved ckpts to:", ckpt_dir)
    print(list(ckpt_dir.glob("*.ckpt"))[:5])

    manual = ckpt_dir / "manual_last.ckpt"
    trainer.save_checkpoint(str(manual))
    print("Manual checkpoint:", manual)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--data", default="configs/data_config.yaml")
    ap.add_argument("--init_from", default="", help="Warm-start weights from a Phase1 ckpt (path)")
    args = ap.parse_args()
    main(args.cfg, args.data)
