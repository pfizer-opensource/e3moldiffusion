import os
import pickle
import lmdb
from torch.utils.data import Dataset
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from experiments.data.abstract_dataset import (
    AbstractDataModuleLigand,
)
from experiments.data.utils import load_pickle, make_splits
from torch.utils.data import Subset

FOLLOW_BATCH = (
    "protein_element",
    "ligand_element",
    "ligand_bond_type",
)
from tqdm.auto import tqdm
from experiments.data.crossdocked.utils import (
    PDBProtein,
    parse_sdf_file,
    torchify_dict,
    get_batch_connectivity_matrix,
)


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

        self.idx_train, self.idx_val, self.idx_test = make_splits(
            len(self.dataset),
            train_size=hparams.train_size,
            val_size=hparams.val_size,
            test_size=hparams.test_size,
            seed=hparams.seed,
            filename=os.path.join(self.hparams["save_dir"], "splits.npz"),
            splits=None,
        )
        print(
            f"train {len(self.idx_train)}, val {len(self.idx_val)}, test {len(self.idx_test)}"
        )
        train_dataset = Subset(self.dataset, self.idx_train)
        val_dataset = Subset(self.dataset, self.idx_val)
        test_dataset = Subset(self.dataset, self.idx_test)

        self.statistics = {
            "train": train_dataset.statistics,
            "val": val_dataset.statistics,
            "test": test_dataset.statistics,
        }
        super().__init__(hparams, train_dataset, val_dataset, test_dataset)

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
