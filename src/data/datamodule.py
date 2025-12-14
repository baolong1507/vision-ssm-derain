from pathlib import Path
from torch.utils.data import DataLoader, ConcatDataset
from .datasets import PairedDerainDataset

class DerainDataModule:
    def __init__(self, data_cfg: dict, batch_size: int):
        self.cfg = data_cfg
        self.batch_size = batch_size

    def _ds_one(self, dataset_name: str, split_key: str, train: bool):
        root = Path(self.cfg["data_root"]) / dataset_name
        sub = self.cfg["subdirs"]
        split = sub[split_key]
        return PairedDerainDataset(
            root=root,
            split=split,
            rain_dir=sub["rain"],
            gt_dir=sub["gt"],
            crop_size=self.cfg["crop_size"],
            train=train,
        )

    def train_loader(self):
        dss = [self._ds_one(d, "train", True) for d in self.cfg["datasets"]]
        ds = ConcatDataset(dss) if len(dss) > 1 else dss[0]
        return DataLoader(ds, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.cfg["num_workers"], pin_memory=self.cfg["pin_memory"])

    def val_loader(self):
        dss = [self._ds_one(d, "val", False) for d in self.cfg["datasets"]]
        ds = ConcatDataset(dss) if len(dss) > 1 else dss[0]
        return DataLoader(ds, batch_size=1, shuffle=False,
                          num_workers=self.cfg["num_workers"], pin_memory=self.cfg["pin_memory"])

    def test_loader(self, dataset_name: str):
        ds = self._ds_one(dataset_name, "test", False)
        return DataLoader(ds, batch_size=1, shuffle=False,
                          num_workers=self.cfg["num_workers"], pin_memory=self.cfg["pin_memory"])
