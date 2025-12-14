from pathlib import Path
from typing import List, Tuple, Dict, Optional
import re
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

def canonical_stem(st: str) -> str:
    """
    Chuẩn hóa tên file để ghép cặp:
    - Rain1400: '222_1'..'222_14' -> key '222', '222_clean' -> key '222'
    - Các bộ khác: bỏ hậu tố _rain/_rainy/_clean nếu có.
    """
    m = re.match(r'^(\d+)_\d{1,2}$', st)
    if m:
        return m.group(1)
    st = re.sub(r'_(rain|rainy|clean)$', '', st, flags=re.IGNORECASE)
    return st

def _list_imgs(p: Path) -> List[Path]:
    exts = (".png", ".jpg", ".jpeg", ".bmp")
    files = []
    for e in exts:
        files += list(p.glob(f"*{e}"))
    return sorted(files)

def _read_rgb(fp: Path) -> np.ndarray:
    img = cv2.imread(str(fp), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {fp}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

class PairedDerainDataset(Dataset):
    """
    Robust paired dataset:
    - Pair by canonical_stem() instead of exact filename match
    - Works for Rain1400: xxx_k -> xxx_clean / xxx
    - Supports albumentations transform: transform(image=..., image_gt=...)
    """

    def __init__(
        self,
        root=None,
        roots=None,
        split="train",
        # directory names (multiple aliases for compatibility)
        inp_dir=None,
        rain_dir=None,
        input_dir=None,
        gt_dir=None,
        target_dir=None,
        # transforms (multiple aliases)
        transform=None,
        tfms=None,
        transforms=None,
        shuffle=False,
        dataset_name="",
        **kwargs
    ):
        # resolve root
        if roots is not None and isinstance(roots, (list, tuple)) and len(roots) > 0:
            root = roots[0]
        if root is None:
            raise ValueError("PairedDerainDataset: root/roots must be provided.")
        self.root = Path(root)

        self.split = split

        # resolve folder keys
        inp = inp_dir or rain_dir or input_dir or "input"
        gt  = gt_dir  or target_dir or "target"

        self.inp_path = self.root / split / inp
        self.gt_path  = self.root / split / gt

        if not self.inp_path.exists():
            raise FileNotFoundError(f"Input dir not found: {self.inp_path}")
        if not self.gt_path.exists():
            raise FileNotFoundError(f"GT dir not found: {self.gt_path}")

        # resolve transform
        self.transform = transform or tfms or transforms
        self.dataset_name = dataset_name

        inp_files = _list_imgs(self.inp_path)
        gt_files  = _list_imgs(self.gt_path)

        # Build GT map by canonical key, prefer *_clean if exists
        gt_map: Dict[str, Path] = {}
        for g in gt_files:
            key = canonical_stem(g.stem)
            if key not in gt_map:
                gt_map[key] = g
            else:
                # prefer "clean"
                if ("clean" in g.stem.lower()) and ("clean" not in gt_map[key].stem.lower()):
                    gt_map[key] = g

        # Build pairs
        pairs: List[Tuple[Path, Path]] = []
        miss = 0
        for i in inp_files:
            key = canonical_stem(i.stem)
            g = gt_map.get(key, None)
            if g is None:
                miss += 1
                continue
            pairs.append((i, g))

        if len(pairs) == 0:
            raise RuntimeError(f"No paired samples found: {self.inp_path} vs {self.gt_path}")

        if miss > 0:
            print(f"[WARN] {dataset_name or self.root.name} {split}: missing GT for {miss} inputs (after canonical match)")

        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        inp_fp, gt_fp = self.pairs[idx]
        image = _read_rgb(inp_fp)
        image_gt = _read_rgb(gt_fp)

        if self.transform is not None:
            # albumentations style
            try:
                out = self.transform(image=image, image_gt=image_gt)
                x = out["image"]
                y = out["image_gt"]
            except TypeError:
                # fallback if user passes a custom callable
                x, y = self.transform(image, image_gt)
        else:
            # fallback: [0..1] tensor
            x = torch.from_numpy(image).permute(2,0,1).float() / 255.0
            y = torch.from_numpy(image_gt).permute(2,0,1).float() / 255.0

        return {"rain": x, "gt": y, "name": inp_fp.name}
