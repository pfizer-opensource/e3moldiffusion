from typing import Optional

from experiments.data.geom_dataset_adaptive import GeomDrugsDataset
from pytorch_lightning import LightningDataModule
from torch_geometric.data import DataLoader
from tqdm import tqdm

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


class GeomDataModule(LightningDataModule):
    def __init__( self, cfg):
        super().__init__()
        
        self.datadir = cfg.dataset_root
        root_path = cfg.dataset_root
        self.pin_memory = True
        self.persistent_workers = False
        
        train_dataset = GeomDrugsDataset(
            split="train", root=root_path, remove_h=cfg.remove_hs
        )
        val_dataset = GeomDrugsDataset(
            split="val", root=root_path, remove_h=cfg.remove_hs
        )
        test_dataset = GeomDrugsDataset(
            split="test", root=root_path, remove_h=cfg.remove_hs
        )
        self.remove_h = cfg.remove_hs
        self.statistics = {
            "train": train_dataset.statistics,
            "val": val_dataset.statistics,
            "test": test_dataset.statistics,
        }
        
    def setup(self, stage: Optional[str] = None) -> None:
        train_dataset = GeomDrugsDataset(root=self.cfg.dataset_root, split="train", remove_h=self.cfg.remove_hs)
        val_dataset = GeomDrugsDataset(root=self.cfg.dataset_root, split="val", remove_h=self.cfg.remove_hs)
        test_dataset = GeomDrugsDataset(root=self.cfg.dataset_root, split="test", remove_h=self.cfg.remove_hs)

        if stage == "fit" or stage is None:
            self.train_dataset = train_dataset
            self.val_dataset = val_dataset
            self.test_dataset = test_dataset

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


if __name__ == "__main__":
    # Creating the Pytorch Geometric InMemoryDatasets
    DATAROOT = "/sharedhome/let55/projects/e3moldiffusion/experiments/geom/data"
    dataset = GeomDrugsDataset(root=DATAROOT, split="val", remove_h=False)
    print(dataset)
    dataset = GeomDrugsDataset(root=DATAROOT, split="test", remove_h=False)
    print(dataset)
    dataset = GeomDrugsDataset(root=DATAROOT, split="train", remove_h=False)
    print(dataset)
    print(dataset[0])
    print(dataset[0].edge_attr)