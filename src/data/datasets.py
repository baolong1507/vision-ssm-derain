from typing import List, Dict, Tuple, Optional
from pathlib import Path
from PIL import Image
import re
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from .transforms import to_numpy, random_crop_pair, hflip_pair

def canonical_stem(st: str) -> str:
    """
    Chuẩn hóa tên file để ghép cặp:
    - Rain1400: '1_1'..'1_14' -> key '1', '1_clean' -> key '1'
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
        raise FileNotFoundError(f"Cannot read: {fp}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

class PairedDerainDataset(Dataset):
    def __init__(self, root: str | Path, split: str, rain_dir="rain", gt_dir="gt",
                 crop_size=256, train=True):
        self.root = Path(root)
        self.split = split
        self.rain_path = self.root / split / rain_dir
        self.gt_path = self.root / split / gt_dir
        self.crop_size = crop_size
        self.train = train

        self.rain_files = sorted(list(self.rain_path.glob("*.png")) + list(self.rain_path.glob("*.jpg")))
        if len(self.rain_files) == 0:
            raise FileNotFoundError(f"No images found in: {self.rain_path}")

    def __len__(self):
        return len(self.rain_files)

    def __getitem__(self, idx):
        rain_fp = self.rain_files[idx]
        gt_fp = self.gt_path / rain_fp.name
        if not gt_fp.exists():
            # fallback: some datasets use different ext
            alt = list(self.gt_path.glob(rain_fp.stem + ".*"))
            if len(alt) == 0:
                raise FileNotFoundError(f"GT not found for {rain_fp.name} in {self.gt_path}")
            gt_fp = alt[0]

        rain = to_numpy(Image.open(rain_fp).convert("RGB"))
        gt = to_numpy(Image.open(gt_fp).convert("RGB"))

        if self.train:
            rain, gt = random_crop_pair(rain, gt, self.crop_size)
            rain, gt = hflip_pair(rain, gt, p=0.5)

        # HWC -> CHW
        rain = torch.from_numpy(np.transpose(rain, (2, 0, 1))).float()
        gt = torch.from_numpy(np.transpose(gt, (2, 0, 1))).float()
        return {"rain": rain, "gt": gt, "name": rain_fp.name}
