from typing import Optional

from experiments.data.pepconf.pepconf_dataset_adaptive import PepConfDataset
from pytorch_lightning import LightningDataModule
from torch_geometric.data import DataLoader
from tqdm import tqdm
from os.path import join
from experiments.data.utils import train_subset
from torch.utils.data import Subset
from experiments.data.abstract_dataset import AbstractDataModule

full_atom_encoder = {
    "H": 0,
    "B": 1,
    "C": 2,
    "N": 3,
    "O": 4,
    "F": 5,
    "Al": 6,
    "Si": 7,
    "P": 8,
    "S": 9,
    "Cl": 10,
    "As": 11,
    "Br": 12,
    "I": 13,
    "Hg": 14,
    "Bi": 15,
}


class PepConfDataModule(AbstractDataModule):
    def __init__(self, cfg):
        self.datadir = cfg.dataset_root
        root_path = cfg.dataset_root
        self.cfg = cfg
        self.pin_memory = True
        self.persistent_workers = False

        train_dataset = PepConfDataset(
            split="train", root=root_path, remove_h=cfg.remove_hs
        )
        val_dataset = PepConfDataset(
            split="val", root=root_path, remove_h=cfg.remove_hs
        )
        test_dataset = PepConfDataset(
            split="test", root=root_path, remove_h=cfg.remove_hs
        )
        if cfg.select_train_subset:
            self.idx_train = train_subset(
                dset_len=len(train_dataset),
                train_size=cfg.train_size,
                seed=cfg.seed,
                filename=join(cfg.save_dir, "splits.npz"),
            )
            train_dataset = Subset(train_dataset, self.idx_train)

        self.remove_h = cfg.remove_hs
        self.statistics = {
            "train": train_dataset.statistics,
            "val": val_dataset.statistics,
            "test": test_dataset.statistics,
        }
        super().__init__(cfg, train_dataset, val_dataset, test_dataset)

    def train_dataloader(self, shuffle=True):
        dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=self.pin_memory,
            shuffle=shuffle,
            persistent_workers=self.persistent_workers,
        )
        return dataloader

    def val_dataloader(self, shuffle=False):
        dataloader = DataLoader(
            dataset=self.val_dataset,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=self.pin_memory,
            shuffle=shuffle,
            persistent_workers=self.persistent_workers,
        )
        return dataloader

    def test_dataloader(self, shuffle=False):
        dataloader = DataLoader(
            dataset=self.test_dataset,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=self.pin_memory,
            shuffle=shuffle,
            persistent_workers=self.persistent_workers,
        )
        return dataloader

    def get_dataloader(self, dataset, stage):
        if stage == "train":
            batch_size = self.cfg.batch_size
            shuffle = True
        elif stage in ["val", "test"]:
            batch_size = self.cfg.inference_batch_size
            shuffle = False

        dl = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            shuffle=shuffle,
        )

        return dl
