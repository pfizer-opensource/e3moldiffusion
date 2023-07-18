from rdkit import RDLogger
import torch
from os.path import join
from torch_geometric.data import InMemoryDataset, DataLoader
from experiments.data.utils import load_pickle
from experiments.data.abstract_dataset import (
    AbstractAdaptiveDataModule,
)
from experiments.utils import make_splits
from torch.utils.data import Subset


full_atom_encoder = {
    "H": 0,
    "C": 1,
    "N": 2,
    "O": 3,
    "F": 4,
    "Si": 5,
    "P": 6,
    "S": 7,
    "Cl": 8,
    "Br": 9,
    "I": 10,
}


class PubChemDataset(InMemoryDataset):
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

    @property
    def raw_file_names(self):
        return [
            "data_list_0.pickle", 
            "data_list_3.pickle", 
            "data_list_4.pickle", 
            "data_list_5.pickle"
            ]

    @property
    def processed_file_names(self):
        return [
            f"pubchem_data.pt",
        ]

    def download(self):
        raise ValueError(
            "Download and preprocessing is manual. If the data is already downloaded, "
            f"check that the paths are correct. Root dir = {self.root} -- raw files {self.raw_paths}"
        )

    def process(self):
        RDLogger.DisableLog("rdApp.*")
        file_1 = load_pickle(self.raw_paths[0])
        file_2 = load_pickle(self.raw_paths[0])
        file_3 = load_pickle(self.raw_paths[0])
        file_4 = load_pickle(self.raw_paths[0])
        data_list = file_1 + file_2 + file_3 + file_4
        del file_1
        del file_2
        del file_3
        del file_4
        
        torch.save(self.collate(data_list), self.processed_paths[0])


class PubChemDataModule(AbstractAdaptiveDataModule):
    def __init__(self, cfg):
        self.datadir = cfg.dataset_root
        root_path = cfg.dataset_root
        self.pin_memory = True

        dataset = PubChemDataset(
            split="train", root=root_path, remove_h=cfg.remove_hs
        )
        self.idx_train, self.idx_val, self.idx_test = make_splits(
            len(dataset),
            0.9,
            0.1,
            None,
            42,
            join(self.hparams["save_dir"], "splits.npz"),
            None,
        )
        print(
            f"train {len(self.idx_train)}, val {len(self.idx_val)}, test {len(self.idx_test)}"
        )
        train_dataset = Subset(dataset, self.idx_train)
        val_dataset = Subset(dataset, self.idx_val)
        test_dataset = Subset(dataset, self.idx_test)

        self.remove_h = cfg.remove_hs

        super().__init__(cfg, train_dataset, val_dataset, test_dataset)

    def _train_dataloader(self, shuffle=True):
        dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=self.pin_memory,
            shuffle=shuffle,
            persistent_workers=False,
        )
        return dataloader

    def _val_dataloader(self, shuffle=False):
        dataloader = DataLoader(
            dataset=self.val_dataset,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=self.pin_memory,
            shuffle=shuffle,
            persistent_workers=False,
        )
        return dataloader

    def _test_dataloader(self, shuffle=False):
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
        if self.cfg.dataset == "qm9" or self.cfg.dataset == "drugs":
            dataloader = self.get_dataloader(self.train_dataset, "val")
            return self.compute_mean_mad_from_dataloader(dataloader, properties_list)
        elif self.cfg.dataset == "qm9_1half" or self.cfg.dataset == "qm9_2half":
            dataloader = self.get_dataloader(self.val_dataset, "val")
            return self.compute_mean_mad_from_dataloader(dataloader, properties_list)
        else:
            raise Exception("Wrong dataset name")

    def compute_mean_mad_from_dataloader(self, dataloader, properties_list):
        property_norms = {}
        for property_key in properties_list:
            try:
                property_name = property_key + "_mm"
                values = getattr(dataloader.dataset[:], property_name)
            except:
                property_name = property_key
                idx = dataloader.dataset[:].label2idx[property_name]
                values = torch.tensor(
                    [data.y[:, idx] for data in dataloader.dataset[:]]
                )

            mean = torch.mean(values)
            ma = torch.abs(values - mean)
            mad = torch.mean(ma)
            property_norms[property_key] = {}
            property_norms[property_key]["mean"] = mean
            property_norms[property_key]["mad"] = mad
            del values
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

