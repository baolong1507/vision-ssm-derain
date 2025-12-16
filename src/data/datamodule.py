from pathlib import Path
import torch
import inspect
from torch.utils.data import DataLoader, ConcatDataset, random_split
from .datasets import PairedDerainDataset
from .transforms_albu import build_transforms as build_tfms_albu
from torch.utils.data import Subset

class DerainDataModule:
    def __init__(self, data_cfg: dict, train_cfg: dict, cfg: dict):
        """
        data_cfg: từ configs/data_config.yaml
        train_cfg: từ phase*.yaml (batch_size, val_ratio, split_seed, auto_split_val, ...)
        """
        if data_cfg is None and cfg is not None:
            data_cfg = cfg
        if data_cfg is None or train_cfg is None:
            raise ValueError("Need data_cfg (or cfg) and train_cfg")
        
        self.dcfg = data_cfg
        self.tcfg = train_cfg

        self.train_ds = None
        self.val_ds = None

    def _tcfg_get(self, key, default=None):
        # ưu tiên train.<key> nếu tồn tại
        try:
            if "train" in self.tcfg and key in self.tcfg.train:
                return self.tcfg.train[key]
        except Exception:
            pass
        # fallback root-level
        try:
            if key in self.tcfg:
                return self.tcfg[key]
        except Exception:
            pass
        return default
    def _has_val_folder(self, dataset_name: str) -> bool:
        root = Path(self.dcfg["data_root"]) / dataset_name
        sub = self.dcfg["subdirs"]
        val_dir = root / sub["val"] / sub["inp"]
        return val_dir.exists()

    def _make_one(self, dataset_name: str, split_key: str, tfms):
        root = Path(self.dcfg["data_root"]) / dataset_name
        sub = self.dcfg["subdirs"]
        split = sub[split_key]

        inp = sub.get("inp", sub.get("rain", "input"))
        gt  = sub.get("gt", "target")

        sig = inspect.signature(PairedDerainDataset.__init__).parameters

        kwargs = {}

        # root style: some code uses roots=[...]
        if "roots" in sig:
            kwargs["roots"] = [str(root)]
        elif "root" in sig:
            kwargs["root"] = root
        else:
            # fallback (rare)
            kwargs["root"] = root

        # split / mode
        if "split" in sig:
            kwargs["split"] = split
        elif "mode" in sig:
            kwargs["mode"] = split
        elif "phase" in sig:
            kwargs["phase"] = split

        # input / gt dir naming
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

        # transforms naming (transform / tfms / transforms)
        if "transform" in sig:
            kwargs["transform"] = tfms
        elif "tfms" in sig:
            kwargs["tfms"] = tfms
        elif "transforms" in sig:
            kwargs["transforms"] = tfms

        # train flags (if supported)
        if "shuffle" in sig:
            kwargs["shuffle"] = (split_key == "train")
        if "is_train" in sig:
            kwargs["is_train"] = (split_key == "train")
        if "train" in sig:
            kwargs["train"] = (split_key == "train")

        # optional dataset name (if supported)
        if "dataset_name" in sig:
            kwargs["dataset_name"] = dataset_name
        if "name" in sig:
            kwargs["name"] = dataset_name

        # IMPORTANT: filter out anything not in signature (extra safety)
        kwargs = {k: v for k, v in kwargs.items() if k in sig}

        return PairedDerainDataset(**kwargs)

    def setup(self, train_tfms, val_tfms):
        auto_split = bool(self._tcfg_get("auto_split_val", True))
        val_ratio  = float(self._tcfg_get("val_ratio", 0.1))
        split_seed = int(self._tcfg_get("split_seed", 42))

        train_sets, val_sets = [], []
        g = torch.Generator().manual_seed(split_seed)

        for name in self.dcfg["datasets"]:
            # luôn tạo full train
            full_train = self._make_one(name, "train", train_tfms)

            # nếu có val folder thật -> dùng
            if self._has_val_folder(name) and not auto_split:
                val_ds = self._make_one(name, "val", val_tfms)
                train_sets.append(full_train)
                val_sets.append(val_ds)
                print(f"[{name}] Using existing val folder.")
                continue

            # nếu không có val -> random split
            base = self._make_one(name, "train", tfms=None)
            n = len(base)
            n_val = max(1, int(val_ratio * n))

            # 2) split indices (seed cố định)
            idx = torch.randperm(n, generator=g).tolist()
            val_idx = idx[:n_val]
            train_idx = idx[n_val:]

            # 3) tạo 2 dataset thật với transform khác nhau
            train_full = self._make_one(name, "train", train_tfms)
            val_full   = self._make_one(name, "train", val_tfms)

            train_sets.append(Subset(train_full, train_idx))
            val_sets.append(Subset(val_full, val_idx))

            print(f"[{name}] Auto-split: val={n_val}/{n} (seed={split_seed})")

        self.train_ds = ConcatDataset(train_sets) if len(train_sets) > 1 else train_sets[0]
        self.val_ds   = ConcatDataset(val_sets)   if len(val_sets) > 1 else val_sets[0]

    def build_transforms(self):
        img_size  = int(self.dcfg.get("img_size", 256)) if isinstance(self.dcfg, dict) else 256
        crop_size = int(self.dcfg.get("crop_size", 256)) if isinstance(self.dcfg, dict) else 256

        # nếu bạn để img_size/crop_size trong train_cfg (phase yaml) thì ưu tiên train_cfg
        img_size  = int(self.tcfg.get("img_size", img_size))
        crop_size = int(self.tcfg.get("crop_size", crop_size))

        train_tfms = build_tfms_albu(img_size, crop_size, True)
        val_tfms   = build_tfms_albu(img_size, crop_size, False)

        return train_tfms, val_tfms
    
    def train_loader(self):
        return DataLoader(
            self.train_ds,
            batch_size = int(self._tcfg_get("batch_size", 8)),
            shuffle=True,
            num_workers=int(self.dcfg["num_workers"]),
            pin_memory=bool(self.dcfg["pin_memory"]),
        )

    def val_loader(self):
        return DataLoader(
            self.val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=int(self.dcfg["num_workers"]),
            pin_memory=bool(self.dcfg["pin_memory"]),
        )

    def test_loader(self, dataset_name: str, val_tfms):
        ds = self._make_one(dataset_name, "test", val_tfms)
        return DataLoader(
            ds, batch_size=1, shuffle=False,
            num_workers=int(self.dcfg["num_workers"]),
            pin_memory=bool(self.dcfg["pin_memory"]),
        )
    
    def train_dataloader(self):
        return self.train_loader()

    def val_dataloader(self):
        return self.val_loader()