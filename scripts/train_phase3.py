import argparse
from omegaconf import OmegaConf
from src.utils.seed import seed_everything
from src.data.datamodule import DerainDataModule
from src.models.fessm_net import FESSMNet
from src.lit_module import LitDerain
from src.utils.io import ensure_dir
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

def main(cfg_path, data_cfg_path):
    cfg = OmegaConf.load(cfg_path)
    data_cfg = OmegaConf.load(data_cfg_path)

    seed_everything(int(cfg.seed))

    dm = DerainDataModule(data_cfg, batch_size=int(cfg.train.batch_size))
    train_loader = dm.train_loader()
    val_loader = dm.val_loader()

    model = FESSMNet(
        base_ch=int(cfg.model.base_ch),
        freq_ch=int(cfg.model.freq_ch),
        ssm_mode=str(cfg.model.ssm_mode),
    )
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
