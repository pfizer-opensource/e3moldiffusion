import os
import pickle

import experiments.data.utils as dataset_utils
import lmdb
import numpy as np
import torch
from experiments.data.abstract_dataset import (
    AbstractDataModuleLigand,
)
from experiments.data.crossdocked.utils import (
    PDBProtein,
    parse_sdf_file,
    torchify_dict,
)
from experiments.data.metrics import compute_all_statistics
from experiments.data.utils import (
    load_pickle,
    save_pickle,
)
from torch.utils.data import Dataset, Subset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

FOLLOW_BATCH = (
    "protein_element",
    "ligand_element",
    "ligand_bond_type",
)
from tqdm.auto import tqdm

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
atom_decoder = {v: k for k, v in full_atom_encoder.items()}


class PocketLigandPairDataset(Dataset):
    def __init__(
        self, root, split="train", remove_hs=False, transform=None, version="final"
    ):
        super().__init__()
        self.index_path = os.path.join(root, "index.pkl")
        self.processed_path = os.path.join(
            root, "crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb"
        )
        self.transform = transform
        self.db = None

        self.keys = None

        if not os.path.exists(self.processed_path):
            print(f"{self.processed_path} does not exist, begin processing data")
            self._process()

        self.statistics = dataset_utils.Statistics(
            num_nodes=load_pickle(self.processed_paths[1]),
            atom_types=torch.from_numpy(np.load(self.processed_paths[2])),
            bond_types=torch.from_numpy(np.load(self.processed_paths[3])),
            charge_types=torch.from_numpy(np.load(self.processed_paths[4])),
            valencies=load_pickle(self.processed_paths[5]),
            bond_lengths=load_pickle(self.processed_paths[6]),
            bond_angles=torch.from_numpy(np.load(self.processed_paths[7])),
            is_aromatic=torch.from_numpy(np.load(self.processed_paths[8])).float(),
            is_in_ring=torch.from_numpy(np.load(self.processed_paths[9])).float(),
            hybridization=torch.from_numpy(np.load(self.processed_paths[10])).float(),
        )
        self.smiles = load_pickle(self.processed_paths[11])

    def _connect_db(self):
        """
        Establish read-only database connection
        """
        assert self.db is None, "A connection has already been opened."
        self.db = lmdb.open(
            self.processed_path,
            map_size=10 * (1024 * 1024 * 1024),  # 10GB
            create=False,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.db.begin() as txn:
            self.keys = list(txn.cursor().iternext(values=False))

    def _close_db(self):
        self.db.close()
        self.db = None
        self.keys = None

    def _process(self):
        db = lmdb.open(
            self.processed_path,
            map_size=10 * (1024 * 1024 * 1024),  # 10GB
            create=True,
            subdir=False,
            readonly=False,  # Writable
        )
        with open(self.index_path, "rb") as f:
            index = pickle.load(f)

        num_skipped = 0
        with db.begin(write=True, buffers=True) as txn:
            for i, (pocket_fn, ligand_fn, *_) in enumerate(tqdm(index)):
                if pocket_fn is None:
                    continue
                try:
                    # data_prefix = '/data/work/jiaqi/binding_affinity'
                    data_prefix = self.raw_path
                    pocket_dict = PDBProtein(
                        os.path.join(data_prefix, pocket_fn)
                    ).to_dict_atom()
                    ligand_dict = parse_sdf_file(os.path.join(data_prefix, ligand_fn))
                    data = ProteinLigandData.from_protein_ligand_dicts(
                        protein_dict=torchify_dict(pocket_dict),
                        ligand_dict=torchify_dict(ligand_dict),
                    )
                    data.protein_filename = pocket_fn
                    data.ligand_filename = ligand_fn
                    data = data.to_dict()  # avoid torch_geometric version issue
                    txn.put(key=str(i).encode(), value=pickle.dumps(data))
                except:
                    num_skipped += 1
                    print(
                        "Skipping (%d) %s"
                        % (
                            num_skipped,
                            ligand_fn,
                        )
                    )
                    continue
        db.close()

    def __len__(self):
        if self.db is None:
            self._connect_db()
        return len(self.keys)

    def __getitem__(self, idx):
        data = self.get_ori_data(idx)
        if self.transform is not None:
            data = self.transform(data)
        return data

    def get_ori_data(self, idx):
        if self.db is None:
            self._connect_db()
        key = self.keys[idx]
        data = pickle.loads(self.db.begin().get(key))
        data = ProteinLigandData(**data)
        data.id = idx
        assert data.protein_pos.size(0) > 0
        data.ligand_bond_type = data.edge_attr
        data.ligand_bond_index = data.edge_index
        return data


class ProteinLigandData(Data):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def from_protein_ligand_dicts(protein_dict=None, ligand_dict=None, **kwargs):
        instance = ProteinLigandData(**kwargs)

        if protein_dict is not None:
            for key, item in protein_dict.items():
                instance["protein_" + key] = item

        if ligand_dict is not None:
            for key, item in ligand_dict.items():
                instance["ligand_" + key] = item

        instance["ligand_nbh_list"] = {
            i.item(): [
                j.item()
                for k, j in enumerate(instance.ligand_bond_index[1])
                if instance.ligand_bond_index[0, k].item() == i
            ]
            for i in instance.ligand_bond_index[0]
        }
        return instance

    def __inc__(self, key, value, *args, **kwargs):
        if key == "ligand_bond_index":
            return self["ligand_element"].size(0)
        else:
            return super().__inc__(key, value)


class ProteinLigandDataLoader(DataLoader):
    def __init__(
        self, dataset, batch_size=1, shuffle=False, follow_batch=FOLLOW_BATCH, **kwargs
    ):
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            follow_batch=follow_batch,
            **kwargs,
        )


class PocketLigandDataset(AbstractDataModuleLigand):
    def __init__(self, hparams):
        self.datadir = hparams.dataset_root
        root_path = hparams.dataset_root
        self.pin_memory = True

        self.remove_hs = hparams.remove_hs
        if self.remove_hs:
            print("Pre-Training on dataset with implicit hydrogens")
        self.dataset = PocketLigandPairDataset(
            root=self.datadir, remove_hs=self.remove_hs
        )

        self.train_smiles = self.dataset.smiles

        split_path = os.path.join(root_path, "crossdocked_pocket10_pose_split.pt")
        split = torch.load(split_path)
        subsets = {k: Subset(self.dataset, indices=v) for k, v in split.items()}

        train_dataset = subsets["train"]
        val_dataset = subsets["test"]
        test_dataset = subsets["val"]

        self.statistics = {
            "train": train_dataset.statistics,
            "val": val_dataset.statistics,
            "test": test_dataset.statistics,
        }
        super().__init__(hparams, train_dataset, val_dataset, test_dataset)

    def compute_statistics(self, data):
        statistics = compute_all_statistics(
            data,
            full_atom_encoder,
            charges_dic={-2: 0, -1: 1, 0: 2, 1: 3, 2: 4, 3: 5},
            additional_feats=True,
        )
        save_pickle(statistics.num_nodes, self.processed_paths[1])
        np.save(self.processed_paths[2], statistics.atom_types)
        np.save(self.processed_paths[3], statistics.bond_types)
        np.save(self.processed_paths[4], statistics.charge_types)
        save_pickle(statistics.valencies, self.processed_paths[5])
        save_pickle(statistics.bond_lengths, self.processed_paths[6])
        np.save(self.processed_paths[7], statistics.bond_angles)
        np.save(self.processed_paths[8], statistics.is_aromatic)
        np.save(self.processed_paths[9], statistics.is_in_ring)
        np.save(self.processed_paths[10], statistics.hybridization)

        # save_pickle(set(all_smiles), self.processed_paths[11])

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
