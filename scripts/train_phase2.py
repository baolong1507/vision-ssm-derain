import argparse
from omegaconf import OmegaConf
from src.utils.seed import seed_everything
from src.data.datamodule import DerainDataModule
from src.data.transforms_albu import build_transforms
from src.models.fess_unet import FESSUNet
from src.lit_module import LitDerain
from src.utils.io import ensure_dir
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

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
        precision=cfg.train.precision,
        gradient_clip_val=float(cfg.train.grad_clip),
        log_every_n_steps=int(cfg.train.log_every_n_steps),
        callbacks=[ckpt],
        accelerator="auto",
        devices="auto",
    )
    trainer.fit(lit, train_loader, val_loader)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--data", default="configs/data_config.yaml")
    args = ap.parse_args()
    main(args.cfg, args.data)
