from rdkit import RDLogger
from tqdm import tqdm
import numpy as np
from typing import Optional
import torch
from torch_geometric.data import InMemoryDataset, DataLoader
import experiments.data.utils as dataset_utils
from experiments.data.utils import (
    load_pickle,
    save_pickle,
    get_rdkit_mol,
    write_xyz_file,
)
from experiments.data.metrics import compute_all_statistics
from pytorch_lightning import LightningDataModule

import tempfile
from rdkit import Chem
from torch_geometric.data.collate import collate
from torch_geometric.data.separate import separate
from torch_geometric.data import Data, Dataset
from tqdm import tqdm

full_atom_encoder = {
    "H": 0,
    "C": 1,
    "N": 2,
    "O": 3,
    "F": 4,
    "P": 5,
    "S": 6,
    "Cl": 7,
}
mol_properties = [
    "DIP",
    "HLgap",
    "eAT",
    "eC",
    "eEE",
    "eH",
    "eKIN",
    "eKSE",
    "eL",
    "eNE",
    "eNN",
    "eMBD",
    "eTS",
    "eX",
    "eXC",
    "eXX",
    "mPOL",
]

atomic_energies_dict = {
    1: -13.643321054,
    6: -1027.610746263,
    7: -1484.276217092,
    8: -2039.751675679,
    9: -3139.751675679,
    15: -9283.015861995,
    16: -10828.726222083,
    17: -12516.462339357,
}
atomic_numbers = [1, 6, 7, 8, 9, 15, 16, 17]
convert_z_to_x = {k: i for i, k in enumerate(atomic_numbers)}


class AQMDataset(InMemoryDataset):
    def __init__(
        self, split, root, remove_h, transform=None, pre_transform=None, pre_filter=None
    ):
        assert split in ["train", "val", "test"]
        self.split = split
        self.remove_h = remove_h

        self.atom_encoder = full_atom_encoder
        if remove_h:
            self.atom_encoder = {
                k: v - 1 for k, v in self.atom_encoder.items() if k != "H"
            }

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.statistics = dataset_utils.Statistics(
            num_nodes=load_pickle(self.processed_paths[1]),
            atom_types=torch.from_numpy(np.load(self.processed_paths[2])),
            bond_types=torch.from_numpy(np.load(self.processed_paths[3])),
            charge_types=torch.from_numpy(np.load(self.processed_paths[4])),
            valencies=load_pickle(self.processed_paths[5]),
            bond_lengths=load_pickle(self.processed_paths[6]),
            bond_angles=torch.from_numpy(np.load(self.processed_paths[7])),
            is_aromatic=torch.from_numpy(np.load(self.processed_paths[9])).float(),
            is_in_ring=torch.from_numpy(np.load(self.processed_paths[10])).float(),
            hybridization=torch.from_numpy(np.load(self.processed_paths[11])).float(),
        )
        self.smiles = load_pickle(self.processed_paths[8])

    @property
    def raw_file_names(self):
        if self.split == "train":
            return ["train_data.pickle"]
        elif self.split == "val":
            return ["val_data.pickle"]
        else:
            return ["test_data.pickle"]

    def processed_file_names(self):
        h = "noh" if self.remove_h else "h"
        if self.split == "train":
            return [
                f"train_{h}.pt",
                f"train_n_{h}.pickle",
                f"train_atom_types_{h}.npy",
                f"train_bond_types_{h}.npy",
                f"train_charges_{h}.npy",
                f"train_valency_{h}.pickle",
                f"train_bond_lengths_{h}.pickle",
                f"train_angles_{h}.npy",
                "train_smiles.pickle",
                f"train_is_aromatic_{h}.npy",
                f"train_is_in_ring_{h}.npy",
                f"train_hybridization_{h}.npy",
            ]
        elif self.split == "val":
            return [
                f"val_{h}.pt",
                f"val_n_{h}.pickle",
                f"val_atom_types_{h}.npy",
                f"val_bond_types_{h}.npy",
                f"val_charges_{h}.npy",
                f"val_valency_{h}.pickle",
                f"val_bond_lengths_{h}.pickle",
                f"val_angles_{h}.npy",
                "val_smiles.pickle",
                f"val_is_aromatic_{h}.npy",
                f"val_is_in_ring_{h}.npy",
                f"val_hybridization_{h}.npy",
            ]
        else:
            return [
                f"test_{h}.pt",
                f"test_n_{h}.pickle",
                f"test_atom_types_{h}.npy",
                f"test_bond_types_{h}.npy",
                f"test_charges_{h}.npy",
                f"test_valency_{h}.pickle",
                f"test_bond_lengths_{h}.pickle",
                f"test_angles_{h}.npy",
                "test_smiles.pickle",
                f"test_is_aromatic_{h}.npy",
                f"test_is_in_ring_{h}.npy",
                f"test_hybridization_{h}.npy",
            ]

    def download(self):
        raise ValueError(
            "Download and preprocessing is manual. If the data is already downloaded, "
            f"check that the paths are correct. Root dir = {self.root} -- raw files {self.raw_paths}"
        )

    def process(self):
        RDLogger.DisableLog("rdApp.*")
        all_data = load_pickle(self.raw_paths[0])

        data_list = []
        all_smiles = []
        for i, data in enumerate(tqdm(all_data)):
            data_list.append(data)
            all_smiles.append(data.smiles)

        torch.save(self.collate(data_list), self.processed_paths[0])

        statistics = compute_all_statistics(
            data_list,
            self.atom_encoder,
            charges_dic={-1: 0, 0: 1, 1: 2},
            additional_feats=True,
        )
        save_pickle(statistics.num_nodes, self.processed_paths[1])
        np.save(self.processed_paths[2], statistics.atom_types)
        np.save(self.processed_paths[3], statistics.bond_types)
        np.save(self.processed_paths[4], statistics.charge_types)
        save_pickle(statistics.valencies, self.processed_paths[5])
        save_pickle(statistics.bond_lengths, self.processed_paths[6])
        np.save(self.processed_paths[7], statistics.bond_angles)
        save_pickle(set(all_smiles), self.processed_paths[8])

        np.save(self.processed_paths[9], statistics.is_aromatic)
        np.save(self.processed_paths[10], statistics.is_in_ring)
        np.save(self.processed_paths[11], statistics.hybridization)


class AQMDataModule(LightningDataModule):
    def __init__(self, cfg):
        self.datadir = cfg.dataset_root
        root_path = cfg.dataset_root
        self.pin_memory = True

        self.label2idx = {k: i for i, k in enumerate(mol_properties)}

        train_dataset = AQMDataset(
            split="train", root=root_path, remove_h=cfg.remove_hs
        )
        val_dataset = AQMDataset(split="val", root=root_path, remove_h=cfg.remove_hs)
        test_dataset = AQMDataset(split="test", root=root_path, remove_h=cfg.remove_hs)
        self.remove_h = cfg.remove_hs
        self.statistics = {
            "train": train_dataset.statistics,
            "val": val_dataset.statistics,
            "test": test_dataset.statistics,
        }

    def setup(self, stage: Optional[str] = None) -> None:
        train_dataset = AQMDataset(
            root=self.cfg.dataset_root, split="train", remove_h=self.cfg.remove_hs
        )
        val_dataset = AQMDataset(
            root=self.cfg.dataset_root, split="val", remove_h=self.cfg.remove_hs
        )
        test_dataset = AQMDataset(
            root=self.cfg.dataset_root, split="test", remove_h=self.cfg.remove_hs
        )

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
            persistent_workers=False,
        )
        return dataloader

    def val_dataloader(self, shuffle=False):
        dataloader = DataLoader(
            dataset=self.val_dataset,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=self.pin_memory,
            shuffle=shuffle,
            persistent_workers=False,
        )
        return dataloader

    def test_dataloader(self, shuffle=False):
        dataloader = DataLoader(
            dataset=self.test_dataset,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=self.pin_memory,
            shuffle=shuffle,
            persistent_workers=False,
        )
        return dataloader

    def compute_mean_mad(self, properties_list):
        if self.hparams["dataset"] == "aqm":
            dataloader = self.get_dataloader(self.train_dataset, "val")
            return self.compute_mean_mad_from_dataloader(dataloader, properties_list)
        elif (
            self.hparams["dataset"] == "aqm_half"
            or self.hparams["dataset"] == "aqm_2half"
        ):
            dataloader = self.get_dataloader(self.val_dataset, "val")
            return self.compute_mean_mad_from_dataloader(dataloader, properties_list)
        else:
            raise Exception("Wrong dataset name")

    def compute_mean_mad_from_dataloader(self, properties_list):
        property_norms = {}
        for property_key in properties_list:
            idx = self.label2idx[property_key]

            values = self.dataset.data.y[:, idx]
            train_idx = [int(i) for i in self.idx_train]
            values_train = torch.tensor(
                [v for i, v in enumerate(values) if i in train_idx]
            )
            mean = torch.mean(values_train)
            ma = torch.abs(values_train - mean)
            mad = torch.mean(ma)
            property_norms[property_key] = {}
            property_norms[property_key]["mean"] = mean
            property_norms[property_key]["mad"] = mad
            del values_train
        return property_norms

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
