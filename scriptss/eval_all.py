import argparse
import csv
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torchmetrics.functional.image import peak_signal_noise_ratio, structural_similarity_index_measure

from src.data.datamodule import DerainDataModule
from src.data.transforms_albu import build_transforms
from src.models.factory import build_model


def _cfg_get(cfg_node, key, default=None):
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


def _extract_xy(batch):
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


@torch.no_grad()
def evaluate_one(cfg_path, data_cfg_path, ckpt_path, split="val", max_batches=None):
    cfg = OmegaConf.load(cfg_path)
    data_cfg = OmegaConf.load(data_cfg_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_size = int(_cfg_get(data_cfg, "img_size", 256))
    crop_size = int(_cfg_get(data_cfg, "crop_size", img_size))

    train_tfms = build_transforms(img_size, crop_size, True)
    val_tfms = build_transforms(img_size, crop_size, False)

    dm = DerainDataModule(
        data_cfg=data_cfg,
        train_cfg={
            "batch_size": int(_cfg_get(cfg.train, "batch_size", 8)),
            "auto_split_val": bool(_cfg_get(cfg.train, "auto_split_val", True)),
            "val_ratio": float(_cfg_get(cfg.train, "val_ratio", 0.1)),
            "split_seed": int(_cfg_get(cfg, "seed", 42)),
            "img_size": img_size,
            "crop_size": crop_size,
        },
        cfg=cfg,
    )
    dm.setup(train_tfms, val_tfms)

    if split == "train":
        loader = dm.train_loader()
    else:
        loader = dm.val_loader()

    model = build_model(cfg).to(device)
    model.eval()

    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

    # Lightning checkpoint thường có prefix "model."
    model_state = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            model_state[k[len("model."):]] = v
        elif not k.startswith("loss_") and not k.startswith("metric_"):
            # cho phép fallback nếu checkpoint là state_dict model thuần
            model_state[k] = v

    missing, unexpected = model.load_state_dict(model_state, strict=False)

    psnr_vals = []
    ssim_vals = []
    loss_l1_vals = []

    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break

        x, y = _extract_xy(batch)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        pred = model(x)
        pred = pred.clamp(0.0, 1.0)
        y = y.clamp(0.0, 1.0)

        psnr = peak_signal_noise_ratio(pred, y, data_range=1.0)
        ssim = structural_similarity_index_measure(pred, y, data_range=1.0)
        l1 = torch.mean(torch.abs(pred - y))

        psnr_vals.append(float(psnr.detach().cpu()))
        ssim_vals.append(float(ssim.detach().cpu()))
        loss_l1_vals.append(float(l1.detach().cpu()))

    result = {
        "phase": str(_cfg_get(cfg, "phase", "")),
        "model_name": str(_cfg_get(cfg.model, "name", "")),
        "ckpt_path": str(ckpt_path),
        "split": split,
        "num_batches": len(psnr_vals),
        "psnr_mean": sum(psnr_vals) / max(len(psnr_vals), 1),
        "ssim_mean": sum(ssim_vals) / max(len(ssim_vals), 1),
        "l1_mean": sum(loss_l1_vals) / max(len(loss_l1_vals), 1),
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
    }
    return result


def write_csv(rows, out_csv):
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "phase",
        "model_name",
        "split",
        "num_batches",
        "psnr_mean",
        "ssim_mean",
        "l1_mean",
        "missing_keys",
        "unexpected_keys",
        "ckpt_path",
    ]

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def write_md(rows, out_md):
    out_md = Path(out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("| Phase | Model | Split | Batches | PSNR | SSIM | L1 | Missing | Unexpected |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| {r['phase']} | {r['model_name']} | {r['split']} | {r['num_batches']} | "
            f"{r['psnr_mean']:.4f} | {r['ssim_mean']:.4f} | {r['l1_mean']:.6f} | "
            f"{r['missing_keys']} | {r['unexpected_keys']} |"
        )

    out_md.write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="configs/data_config.yaml")
    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--max_batches", type=int, default=None)

    ap.add_argument("--cfg1", default="configs/phase1_unet.yaml")
    ap.add_argument("--ckpt1", required=True)

    ap.add_argument("--cfg2", default="configs/phase2_unet_freq.yaml")
    ap.add_argument("--ckpt2", required=True)

    ap.add_argument("--cfg3", default="configs/phase3_unet_freq_symscan.yaml")
    ap.add_argument("--ckpt3", required=True)

    ap.add_argument("--out_csv", default="outputs/eval_all_results.csv")
    ap.add_argument("--out_md", default="outputs/eval_all_results.md")

    args = ap.parse_args()

    rows = []
    rows.append(evaluate_one(args.cfg1, args.data, args.ckpt1, split=args.split, max_batches=args.max_batches))
    rows.append(evaluate_one(args.cfg2, args.data, args.ckpt2, split=args.split, max_batches=args.max_batches))
    rows.append(evaluate_one(args.cfg3, args.data, args.ckpt3, split=args.split, max_batches=args.max_batches))

    write_csv(rows, args.out_csv)
    write_md(rows, args.out_md)

    print("Saved:", args.out_csv)
    print("Saved:", args.out_md)
    for r in rows:
        print(
            f"[{r['phase']}] {r['model_name']} | "
            f"PSNR={r['psnr_mean']:.4f} | SSIM={r['ssim_mean']:.4f} | "
            f"L1={r['l1_mean']:.6f} | batches={r['num_batches']}"
        )


if __name__ == "__main__":
    main()