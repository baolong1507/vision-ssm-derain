from pathlib import Path
import torch
import inspect
from torch.utils.data import DataLoader, ConcatDataset, random_split
from .datasets import PairedDerainDataset

class DerainDataModule:
    def __init__(self, data_cfg: dict, train_cfg: dict):
        """
        data_cfg: từ configs/data_config.yaml
        train_cfg: từ phase*.yaml (batch_size, val_ratio, split_seed, auto_split_val, ...)
        """
        self.dcfg = data_cfg
        self.tcfg = train_cfg

        self.train_ds = None
        self.val_ds = None

    def _has_val_folder(self, dataset_name: str) -> bool:
        root = Path(self.dcfg["data_root"]) / dataset_name
        sub = self.dcfg["subdirs"]
        val_dir = root / sub["val"] / sub["inp"]
        return val_dir.exists()

    def _make_one(self, dataset_name: str, split_key: str, transform):
        root = Path(self.dcfg["data_root"]) / dataset_name
        sub = self.dcfg["subdirs"]
        split = sub[split_key]

        inp = sub.get("inp", sub.get("rain", "input"))
        gt  = sub.get("gt", "target")

        sig = inspect.signature(PairedDerainDataset.__init__).parameters

        # Case A: dataset class mới dùng inp_dir/gt_dir
        if "inp_dir" in sig:
            return PairedDerainDataset(
                root=root,
                split=split,
                inp_dir=inp,
                gt_dir=gt,
                transform=transform,
                dataset_name=dataset_name,
            )

        # Case B: dataset class cũ dùng rain_dir/gt_dir
        if "rain_dir" in sig:
            return PairedDerainDataset(
                root=root,
                split=split,
                rain_dir=inp,      # input folder
                gt_dir=gt,
                transform=transform,
                dataset_name=dataset_name,
            )

        # Case C: dataset class cũ hơn nữa dùng input_dir/target_dir
        if "input_dir" in sig:
            return PairedDerainDataset(
                root=root,
                split=split,
                input_dir=inp,
                target_dir=gt,
                transform=transform,
                dataset_name=dataset_name,
            )

        raise TypeError(f"Unsupported PairedDerainDataset signature: {list(sig.keys())}")

    def setup(self, train_tfms, val_tfms):
        auto_split = bool(self.tcfg.get("auto_split_val", True))
        val_ratio  = float(self.tcfg.get("val_ratio", 0.1))
        split_seed = int(self.tcfg.get("split_seed", 42))

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
            n = len(full_train)
            n_val = max(1, int(val_ratio * n))
            n_train = n - n_val
            train_sub, val_sub = random_split(full_train, [n_train, n_val], generator=g)
            train_sets.append(train_sub)
            val_sets.append(val_sub)
            print(f"[{name}] Auto-split: val={n_val}/{n} (seed={split_seed})")

        self.train_ds = ConcatDataset(train_sets) if len(train_sets) > 1 else train_sets[0]
        self.val_ds   = ConcatDataset(val_sets)   if len(val_sets) > 1 else val_sets[0]

    def train_loader(self):
        return DataLoader(
            self.train_ds,
            batch_size=int(self.tcfg["batch_size"]),
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
