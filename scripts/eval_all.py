import argparse
import json
import math
import re
from pathlib import Path
import inspect

import torch
from torch.utils.data import DataLoader

from omegaconf import OmegaConf

# --- project imports (assumes PYTHONPATH points to repo root) ---
from src.lit_module import LitDerain
from src.utils.io import ensure_dir, save_json
from src.metrics.psnr import psnr as psnr_fn
from src.losses.ssim import ssim as ssim_fn

from src.data.datasets import PairedDerainDataset
from src.data.transforms_albu import build_transforms

from src.models.unet_baseline import UNetBaseline
from src.models.fess_unet import FESSUNet
from src.models.fessm_net import FESSMNet


def pick_ckpt(ckpt_dir: Path, prefer: str = "best") -> Path:
    """
    prefer:
      - "best": pick epoch*-psnr*.ckpt with highest psnr
      - "last": pick last.ckpt
      - path: direct path
    """
    if prefer and prefer not in ("best", "last"):
        p = Path(prefer)
        if p.exists():
            return p
        raise FileNotFoundError(f"Checkpoint not found: {p}")

    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"ckpt_dir not found: {ckpt_dir}")

    if prefer == "last":
        last = ckpt_dir / "last.ckpt"
        if last.exists():
            return last
        raise FileNotFoundError(f"last.ckpt not found in {ckpt_dir}")

    # prefer == "best"
    cands = sorted(ckpt_dir.glob("*.ckpt"))
    scored = []
    for p in cands:
        m = re.search(r"psnr([0-9]+\.[0-9]+)", p.name)
        if m:
            scored.append((float(m.group(1)), p))
    if scored:
        scored.sort(key=lambda x: x[0])
        return scored[-1][1]

    # fallback to last
    last = ckpt_dir / "last.ckpt"
    if last.exists():
        return last
    raise FileNotFoundError(f"No suitable ckpt found in {ckpt_dir}")


def build_model_from_cfg(cfg):
    name = str(cfg.model.name).lower()
    base_ch = int(cfg.model.get("base_ch", 48))

    if "unet" in name and "fess" not in name:
        return UNetBaseline(base_ch=base_ch)

    if "fess_unet" in name or ("fess" in name and "unet" in name):
        freq_ch = int(cfg.model.get("freq_ch", 16))
        return FESSUNet(base_ch=base_ch, freq_ch=freq_ch)

    if "fessm" in name:
        freq_ch = int(cfg.model.get("freq_ch", 16))
        ssm_mode = str(cfg.model.get("ssm_mode", "convscan"))
        return FESSMNet(base_ch=base_ch, freq_ch=freq_ch, ssm_mode=ssm_mode)

    raise ValueError(f"Unknown model name in cfg.model.name: {cfg.model.name}")


def make_test_dataset(data_cfg, dataset_name: str, tfms):
    """
    Create PairedDerainDataset for <data_root>/<dataset>/<test>/<input|target>
    Uses signature-adaptive kwargs so it works even if your dataset class init differs.
    """
    root = Path(data_cfg.data_root) / dataset_name
    sub = data_cfg.subdirs
    split = sub.test
    inp = sub.inp
    gt = sub.gt

    sig = inspect.signature(PairedDerainDataset.__init__).parameters
    kwargs = {}

    # root / roots
    if "root" in sig:
        kwargs["root"] = root
    elif "roots" in sig:
        kwargs["roots"] = [str(root)]

    # split / mode / phase
    if "split" in sig:
        kwargs["split"] = split
    elif "mode" in sig:
        kwargs["mode"] = split
    elif "phase" in sig:
        kwargs["phase"] = split

    # input/gt dirs
    if "inp_dir" in sig:
        kwargs["inp_dir"] = inp
    if "rain_dir" in sig:
        kwargs["rain_dir"] = inp
    if "input_dir" in sig:
        kwargs["input_dir"] = inp

    if "gt_dir" in sig:
        kwargs["gt_dir"] = gt
    if "target_dir" in sig:
        kwargs["target_dir"] = gt

    # transforms
    if "transform" in sig:
        kwargs["transform"] = tfms
    elif "tfms" in sig:
        kwargs["tfms"] = tfms
    elif "transforms" in sig:
        kwargs["transforms"] = tfms

    # optional name
    if "dataset_name" in sig:
        kwargs["dataset_name"] = dataset_name
    if "name" in sig:
        kwargs["name"] = dataset_name

    # filter to signature only
    kwargs = {k: v for k, v in kwargs.items() if k in sig}

    return PairedDerainDataset(**kwargs)


@torch.no_grad()
def eval_loader(lit: LitDerain, dl: DataLoader, device: torch.device, num_vis: int, vis_dir: Path, prefix: str):
    lit.eval()
    lit.to(device)

    psnrs = []
    ssims = []

    saved = 0
    for i, batch in enumerate(dl):
        rain = batch["rain"].to(device)
        gt = batch["gt"].to(device)
        name = batch.get("name", [f"{i:04d}"])[0]

        pred = lit(rain)
        # ensure [0,1] for metrics/visualization
        pred01 = pred.clamp(0, 1)
        gt01 = gt.clamp(0, 1)
        rain01 = rain.clamp(0, 1)

        p = psnr_fn(pred01, gt01).item()
        s = ssim_fn(pred01, gt01).item()
        psnrs.append(p)
        ssims.append(s)

        if saved < num_vis:
            save_triplet_png(rain01[0], pred01[0], gt01[0], vis_dir / f"{prefix}_{saved:02d}_{name}.png")
            saved += 1

    mean_psnr = float(sum(psnrs) / max(1, len(psnrs)))
    mean_ssim = float(sum(ssims) / max(1, len(ssims)))
    return mean_psnr, mean_ssim, len(psnrs)


def tensor_to_uint8_img(x_chw: torch.Tensor):
    # x in [0,1], CHW
    x = x_chw.detach().float().cpu().clamp(0, 1)
    x = (x * 255.0).round().byte()
    # CHW -> HWC
    return x.permute(1, 2, 0).numpy()


def save_triplet_png(inp_chw, pred_chw, gt_chw, out_path: Path):
    from PIL import Image
    inp = tensor_to_uint8_img(inp_chw)
    pred = tensor_to_uint8_img(pred_chw)
    gt = tensor_to_uint8_img(gt_chw)

    # concat horizontally: input | output | gt
    concat = torch.from_numpy(inp)
    # use numpy concat for simplicity
    import numpy as np
    panel = np.concatenate([inp, pred, gt], axis=1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(panel).save(out_path)


def write_csv(rows, out_csv: Path):
    import csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "n", "psnr", "ssim"])
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True, help="configs/phase*.yaml")
    ap.add_argument("--data", required=True, help="configs/data_config.yaml")
    ap.add_argument("--ckpt", default="best", help="'best' | 'last' | /path/to.ckpt")
    ap.add_argument("--num_vis", type=int, default=10, help="num visualization samples per dataset")
    ap.add_argument("--device", default="auto", help="auto|cuda|cpu")
    args = ap.parse_args()

    cfg = OmegaConf.load(args.cfg)
    data_cfg = OmegaConf.load(args.data)

    # device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # transforms for test/val (no random aug)
    test_tfms = build_transforms(int(data_cfg.img_size), int(data_cfg.crop_size), is_train=False)

    # output dirs
    result_dir = Path(str(cfg.output.result_dir))
    vis_dir = result_dir / "vis_samples"
    ensure_dir(result_dir)
    ensure_dir(vis_dir)

    # pick checkpoint
    ckpt_dir = Path(str(cfg.output.ckpt_dir))
    ckpt_path = pick_ckpt(ckpt_dir, args.ckpt)
    print("Using ckpt:", ckpt_path)

    # build model + load lightning module
    model = build_model_from_cfg(cfg)
    lit = LitDerain.load_from_checkpoint(
        str(ckpt_path),
        model=model,
        lr=float(cfg.train.lr),
        weight_decay=float(cfg.train.weight_decay),
        loss_w=dict(cfg.loss),
        map_location="cpu",
    )

    metrics = {
        "phase": str(cfg.phase),
        "ckpt": str(ckpt_path),
        "datasets": {},
        "overall": {},
    }

    csv_rows = []
    total_n = 0
    total_psnr = 0.0
    total_ssim = 0.0

    for dname in list(data_cfg.datasets):
        ds = make_test_dataset(data_cfg, dname, test_tfms)
        dl = DataLoader(
            ds,
            batch_size=1,
            shuffle=False,
            num_workers=int(data_cfg.num_workers),
            pin_memory=bool(data_cfg.pin_memory),
        )

        mean_psnr, mean_ssim, n = eval_loader(
            lit, dl, device=device,
            num_vis=args.num_vis,
            vis_dir=vis_dir,
            prefix=dname
        )

        metrics["datasets"][dname] = {"n": n, "psnr": mean_psnr, "ssim": mean_ssim}
        csv_rows.append({"dataset": dname, "n": n, "psnr": f"{mean_psnr:.4f}", "ssim": f"{mean_ssim:.4f}"})

        total_n += n
        total_psnr += mean_psnr * n
        total_ssim += mean_ssim * n

        print(f"[{dname}] n={n}  PSNR={mean_psnr:.3f}  SSIM={mean_ssim:.4f}")

    if total_n > 0:
        metrics["overall"] = {
            "n": total_n,
            "psnr": float(total_psnr / total_n),
            "ssim": float(total_ssim / total_n),
        }
    else:
        metrics["overall"] = {"n": 0, "psnr": None, "ssim": None}

    # save json/csv
    out_json = result_dir / "test_metrics.json"
    out_csv = result_dir / "test_metrics.csv"
    save_json(metrics, out_json)
    write_csv(csv_rows, out_csv)

    print("\nSaved:")
    print(" -", out_json)
    print(" -", out_csv)
    print(" -", vis_dir)


if __name__ == "__main__":
    main()
