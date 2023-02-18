import os
import os.path as osp
from typing import Callable, Optional, Sequence, Union

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
import gzip
import io
import itertools as it
import json
import multiprocessing as mp
import pickle
from pathlib import Path

import click
import lmdb
import numpy as np
import pandas as pd
import torch
import torch_geometric
from pytorch_lightning import LightningDataModule
from e3moldiffusion.molfeat import get_bond_feature_dims, smiles_or_mol_to_graph
from torch.utils.data import Subset
from torch_sparse import coalesce
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj
from tqdm import tqdm

try:
    PROCESS_PATH = osp.abspath(__file__.__name__) + "/geom"
except:
    PROCESS_PATH = "/home/let55/workspace/projects/e3moldiffusion/geom"

PATH = "/home/let55/workspace/datasets/geom/rdkit_folder"

DB_READ_PATH = "/hpfs/projects/mlcs/e3moldiffusion"

DATA_PATH = osp.join(PROCESS_PATH, "data")
if not osp.exists(DATA_PATH):
    os.makedirs(DATA_PATH)

QM9_summary = json.load(open(osp.join(PATH, "summary_qm9.json")))
DRUGS_summary = json.load(open(osp.join(PATH, "summary_drugs.json")))

QM9_pickles = os.listdir(osp.join(PATH, "qm9"))
QM9_mapping = [osp.splitext(f)[0] for f in QM9_pickles]
QM9_smi_to_id = {s: i for i, s in enumerate(QM9_mapping)}
QM9_id_to_smi = {i: s for s, i in QM9_smi_to_id.items()}
QM9_pickles = [osp.join(osp.join(PATH, "qm9"), f) for f in QM9_pickles]

DRUGS_pickles = os.listdir(osp.join(PATH, "drugs"))
DRUGS_mapping = [osp.splitext(f)[0] for f in DRUGS_pickles]
DRUGS_smi_to_id = {s: i for i, s in enumerate(DRUGS_mapping)}
DRUGS_id_to_smi = {i: s for s, i in DRUGS_smi_to_id.items()}
DRUGS_pickles = [osp.join(osp.join(PATH, "drugs"), f) for f in DRUGS_pickles]

BOND_FEATURE_DIMS = get_bond_feature_dims()
BOND_FEATURE_DIMS = BOND_FEATURE_DIMS[0]

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def smiles_filter(smi: str) -> bool:
    if "." in smi:
        return False
    mol = Chem.MolFromSmiles(smi)
    if mol:
        mol = Chem.AddHs(mol)
    else:
        return False
    N = mol.GetNumAtoms()
    if N < 4:
        return False
    if mol.GetNumBonds() < 4:
        return False

    return True


class GetHigherOrderEdges:
    def __init__(
        self, order: int = 3, num_bond_types: int = get_bond_feature_dims()[0]
    ):
        super().__init__()
        self.order = order
        self.num_bond_types = num_bond_types

    def __call__(self, data: Data):
        adj = to_dense_adj(data.edge_index).squeeze()
        tmp = adj
        higher_order_adjs = [adj]

        edge_indices = []
        edge_attributes = []
        current_adj = adj
        for i in range(self.order - 1):
            tmp = tmp @ adj
            tmp.fill_diagonal_(0.0)
            tmp = torch.where(tmp > 0.0, torch.ones_like(tmp), torch.zeros_like(tmp))
            new_edges = tmp - current_adj
            new_edges = torch.where(
                new_edges > 0.0, torch.ones_like(new_edges), torch.zeros_like(new_edges)
            )

            current_adj += new_edges
            higher_order_adjs.append(tmp)

            new_edge_index = new_edges.nonzero().T
            new_edge_attr = torch.ones(new_edge_index.size(-1), dtype=torch.long) * (
                self.num_bond_types + i
            )
            edge_indices.append(new_edge_index)
            edge_attributes.append(new_edge_attr)

        edge_indices = torch.cat(edge_indices, dim=-1)
        edge_attributes = torch.cat(edge_attributes, dim=0)

        edge_index_final = torch.cat([data.edge_index, edge_indices], dim=-1)
        edge_attr_final = torch.cat([data.edge_attr, edge_attributes], dim=0)

        # sort
        perm = (edge_index_final[0] * data.x.size(0) + edge_index_final[1]).argsort()
        edge_index_final, edge_attr_final = (
            edge_index_final[:, perm],
            edge_attr_final[perm],
        )

        # edge_index_, edge_attr_ = coalesce(edge_index_final, edge_attr_final,
        #                                   m=data.num_nodes, n=data.num_nodes)
        # assert torch.all(edge_index_final == edge_index_)
        # assert torch.all(edge_attr_final == edge_attr_)

        original_edge_index = data.edge_index.clone()
        is_bond_mask = edge_attr_final < self.num_bond_types

        data.edge_index = edge_index_final
        data.edge_attr = edge_attr_final
        data.bond_index = original_edge_index
        data.is_bond = is_bond_mask

        return data


class MolFeaturization:
    def __init__(self, order: int = 3):
        super().__init__()

        self.order = order
        self.higher_order = GetHigherOrderEdges(order=order)

    @classmethod
    def featurize_smiles_or_mol(
        self, smiles_mol: Union[str, Chem.Mol, dict]
    ) -> Optional[Data]:
        if isinstance(smiles_mol, dict):
            mol = smiles_mol["mol"]
        elif isinstance(smiles_mol, str):
            smiles = smiles_mol
            mol = Chem.MolFromSmiles(smiles)
            mol = Chem.AddHs(mol)
            if mol is None:
                return None
        elif isinstance(smiles_mol, Chem.Mol):
            mol = smiles_mol

        data = smiles_or_mol_to_graph(smol=mol)

        if mol.GetNumConformers() == 1:
            pos = torch.tensor(mol.GetConformer(0).GetPositions(), dtype=torch.float)
        else:
            pos = None

        # canonical smi
        try:
            smiles = Chem.MolToSmiles(Chem.RemoveHs(mol))
        except:
            smiles = Chem.MolToSmiles(mol)

        data.pos = pos
        data.mol = mol
        data.smiles = smiles

        return data

    @classmethod
    def upper_edge_idx(self, data: Data, order: int):
        N = data.x.size(0)
        E = torch.ones((N, N), dtype=torch.long) * (
            (get_bond_feature_dims()[0] - 1) + order
        )
        E[data.edge_index.t()[:, 0], data.edge_index.t()[:, 1]] = data.edge_attr
        e_triu_ids = torch.triu_indices(N, N, offset=1)
        e_trius = E[e_triu_ids.t()[:, 0], e_triu_ids.t()[:, 1]]
        data.edge_index_upper, data.edge_attr_upper = e_triu_ids, e_trius
        return data

    @classmethod
    def fully_connected_edge_idx(self, data: Data, without_self_loop: bool = True):
        N = data.x.size(0)
        row = torch.arange(N, dtype=torch.long)
        col = torch.arange(N, dtype=torch.long)
        row = row.view(-1, 1).repeat(1, N).view(-1)
        col = col.repeat(N)
        edge_index = torch.stack([row, col], dim=0)
        if without_self_loop:
            mask = edge_index[0] != edge_index[1]
            edge_index = edge_index[:, mask]
        data.edge_index_fc = edge_index
        return data
    
    def __call__(self, smiles_mol: Union[str, Chem.Mol, dict]) -> Data:
        data = self.featurize_smiles_or_mol(smiles_mol=smiles_mol["mol"])
        assert data is not None
        # data = self.higher_order(data)
        data = self.fully_connected_edge_idx(data=data, without_self_loop=True)
        # data = self.upper_edge_idx(data, order=self.order)
        data.energy = torch.tensor([smiles_mol["energy"]], dtype=torch.float32).view(
            -1, 1
        )
        data.boltzmannweight = torch.tensor(
            [smiles_mol["boltzmannweight"]], dtype=torch.float32
        ).view(-1, 1)

        return data


def db_sample_helper(file: str, max_conformers: int = 1000):
    try:
        mol = pickle.load(open(file, "rb"))
        if smiles_filter(mol["smiles"]):
            saved_confs_list = []
            geom_ids = []
            saved_confs = 0
            # sort conformers based on boltzman weights
            all_weights = np.array(
                [c.get("boltzmannweight", -1.0) for c in mol.get("conformers")]
            )
            descend_conf_id = (-all_weights).argsort()
            conf_ids = descend_conf_id[:max_conformers]
            confo_list = [mol["conformers"][i] for i in conf_ids]
            for conf in confo_list:
                if saved_confs > max_conformers:
                    break
                conf_mol = conf["rd_mol"]
                assert conf_mol.GetNumConformers() == 1
                assert (
                    conf_mol.GetConformer(0).GetPositions().shape[0]
                    == conf_mol.GetNumAtoms()
                )
                # create binary object to be saved
                buf = io.BytesIO()
                saves = {
                    "mol": conf_mol,
                    "energy": conf["totalenergy"],
                    "boltzmannweight": conf["boltzmannweight"],
                }
                with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=6) as f:
                    f.write(pickle.dumps(saves))
                compressed = buf.getvalue()
                saved_confs_list.append(compressed)
                saved_confs += 1
                geom_ids.append(conf["geom_id"])
            return {
                "geom_ids": geom_ids,
                "confs": saved_confs_list,
                "smiles": [mol["smiles"]] * len(geom_ids),
            }
        else:
            return None
    except Exception:
        # print(f'Had issues opening the conformers in {file}')
        # print(f'Main dictionary has the keys: {list(mol.keys())}')
        return None


def process_files(
    dataset: str = "drugs",
    max_conformers: int = 1000,
    processes: int = 32,
    chunk_size: int = 2048,
    subchunk: int = 256,
):
    """
    :param dataset:
    :param max_conformers:
    :param processes:
    :param chunk_size:
    :param subchunk:
    :return:
    """
    assert dataset in ["qm9", "drugs"]

    if dataset == "qm9":
        pickles_list = QM9_pickles
    else:
        pickles_list = DRUGS_pickles

    pickles_list = [pickles_list[i] for i in np.arange(len(pickles_list))]
    process_path = osp.join(DATA_PATH, dataset)

    if not osp.exists(process_path):
        os.makedirs(process_path)

    save_path = osp.join(process_path, "database")
    if osp.exists(save_path):
        print("Files have been processed already. Aborting execution")
        return None
    else:
        chunked_list = list(chunks(pickles_list, chunk_size))
        chunked_list = [list(chunks(l, subchunk)) for l in chunked_list]

        print(f"Total number of molecules {len(pickles_list)}.")
        print(f"Processing {len(chunked_list)} chunks each of size {chunk_size}.")

        global_datainfo = pd.DataFrame(columns=["global_id", "smiles", "geom_id"])
        env = lmdb.open(str(save_path), map_size=int(1e13))
        global_id = 0
        id_to_idx = {}

        with env.begin(write=True) as txn:
            for chunklist in tqdm(chunked_list, total=len(chunked_list), desc="Chunks"):
                chunkresult = []
                for datachunk in chunklist:
                    with mp.Pool(processes=processes) as pool:
                        res = pool.starmap(
                            func=db_sample_helper,
                            iterable=zip(
                                datachunk, it.repeat(max_conformers, len(datachunk))
                            ),
                        )
                        res = [r for r in res if r is not None]
                    chunkresult.append(res)

                sub_id_list = []
                geom_ids_sub = []
                confs_sub = []
                smiles_sub = []
                for cr in chunkresult:
                    subgeom_ids = [a["geom_ids"] for a in cr]
                    subgeom_ids = [item for sublist in subgeom_ids for item in sublist]
                    subconfs = [a["confs"] for a in cr]
                    subconfs = [item for sublist in subconfs for item in sublist]
                    subsmiles = [a["smiles"] for a in cr]
                    subsmiles = [item for sublist in subsmiles for item in sublist]
                    geom_ids_sub.append(subgeom_ids)
                    confs_sub.append(subconfs)
                    smiles_sub.append(subsmiles)

                geom_ids_sub = [item for sublist in geom_ids_sub for item in sublist]
                confs_sub = [item for sublist in confs_sub for item in sublist]
                smiles_sub = [item for sublist in smiles_sub for item in sublist]

                assert len(geom_ids_sub) == len(confs_sub) == len(smiles_sub)
                # save
                for gid, conf in zip(geom_ids_sub, confs_sub):
                    sub_id_list.append(global_id)
                    result = txn.put(str(global_id).encode(), conf, overwrite=False)
                    if not result:
                        raise RuntimeError(
                            f"LMDB entry {global_id} in {str(save_path)} "
                            "already exists"
                        )
                    id_to_idx[gid] = global_id
                    global_id += 1

                sub_data_info = pd.DataFrame(columns=["global_id", "smiles", "geom_id"])
                sub_data_info["global_id"] = sub_id_list
                sub_data_info["geom_id"] = geom_ids_sub
                sub_data_info["smiles"] = smiles_sub
                global_datainfo = pd.concat(
                    [global_datainfo, sub_data_info], axis=0
                ).reset_index(drop=True)

            # save final meta data
            txn.put(b"num_examples", str(global_datainfo.shape[0]).encode())
            txn.put(b"id_to_idx", pickle.dumps(id_to_idx))
            txn.put(b"global_info", pickle.dumps(global_datainfo))

        # save global pandas dataframe
        save_file = os.path.dirname(save_path)
        save_file = os.path.join(save_file, "global_info.csv")
        global_datainfo.to_csv(save_file, compression=None)
        print("Finished")


class GeomLMDBDataset(torch_geometric.data.Dataset):
    def __init__(
        self,
        data_file: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        env_in_init: bool = True,
        readonly: bool = True,
    ):
        """
        Constructor
        """
        super().__init__(None, transform, pre_transform)

        self.readonly = readonly
        if type(data_file) is list:
            if len(data_file) != 1:
                raise RuntimeError("Need exactly one filepath for lmdb")
            data_file = data_file[0]

        self.data_file = Path(data_file).absolute()
        if not self.data_file.exists():
            raise FileNotFoundError(self.data_file)

        # here we just init the database to retrieve some meta data
        _env = lmdb.open(
            str(self.data_file),
            readonly=readonly,
            lock=False,
            readahead=False,
            meminit=False,
            create=False,
        )

        with _env.begin(write=False) as txn:
            self._num_examples = int(txn.get(b"num_examples"))
            try:
                self._global_info = pickle.loads(txn.get(b"global_info"))
            except TypeError:
                self._global_info = None

            self._id_to_idx = pickle.loads(txn.get(b"id_to_idx"))

        self._env = _env if env_in_init else None
        self._transform = transform

    def _init_db(self):
        self._env = lmdb.open(
            str(self.data_file),
            readonly=self.readonly,
            lock=False,
            readahead=False,
            meminit=False,
            create=False,
        )

    def len(self) -> int:
        r"""Returns the number of graphs stored in the dataset."""
        return self._num_examples

    def _get(self, id: str):
        """
        Only querying samples based on Geom-identifier
        """
        idx = self.id_to_idx(id)
        return self[idx]

    def id_to_idx(self, id: str):
        """
        Only querying samples based on Geom-identifier
        """
        if id not in self._id_to_idx:
            raise IndexError(id)
        idx = self._id_to_idx[id]
        return idx

    def ids_to_indices(self, ids: Sequence):
        """
        Only querying samples based on Geom-identifier
        """
        return [self.id_to_idx(id) for id in ids]

    def ids(self):
        return list(self._id_to_idx.keys())

    def get(self, index: int):
        # (row) id based querying
        if not 0 <= index < self._num_examples:
            raise IndexError(index)
        if self._env is None:
            self._init_db()

        with self._env.begin(write=False) as txn:
            compressed = txn.get(str(index).encode())
            buf = io.BytesIO(compressed)
            with gzip.GzipFile(fileobj=buf, mode="rb") as f:
                serialized = f.read()
            try:
                item = pickle.loads(serialized)
            except:
                return None
        if self._transform:
            item = self._transform(item)
        return item

    def __len__(self) -> int:
        return self._num_examples


def make_lmdb_dataset(dataset: torch.utils.data.Subset, save_path: str):
    num_examples = len(dataset)
    print(f"{num_examples} examples / conformations")

    if not osp.exists(save_path):
        env = lmdb.open(str(save_path), map_size=int(1e13))
        with env.begin(write=True) as txn:
            try:
                id_to_idx = {}
                i = 0
                for j, data in tqdm(enumerate(dataset), total=num_examples):
                    # save data into lmdb
                    buf = io.BytesIO()
                    with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=6) as f:
                        f.write(pickle.dumps(data))
                    compressed = buf.getvalue()
                    result = txn.put(str(i).encode(), compressed, overwrite=False)
                    if not result:
                        raise RuntimeError(
                            f"LMDB entry {i} in {str(save_path)} " "already exists"
                        )
                    id_to_idx[dataset.indices[j]] = i
                    i += 1
            finally:
                txn.put(b"num_examples", str(i).encode())
                txn.put(b"id_to_idx", pickle.dumps(id_to_idx))
    else:
        print(f"filepath {save_path} already exists.")


def create_splits(dataset: str = "drugs"):
    assert dataset in ["qm9", "drugs"]

    if dataset == "qm9":
        id_to_smi = QM9_id_to_smi
    else:
        id_to_smi = DRUGS_id_to_smi

    process_path = osp.join(DATA_PATH, dataset)
    # load entire dataset
    Geom_dataset = GeomLMDBDataset(
        data_file=osp.join(process_path, "database"), transform=None, env_in_init=True
    )

    global_info = Geom_dataset._global_info
    global_info.smiles = global_info.smiles.astype(str)
    
    unique_smiles = global_info.smiles.unique()
    smiles_to_molID = {smi: i for i, smi in enumerate(unique_smiles)}
    
    smiles_to_id_DF = pd.DataFrame()
    smiles_to_id_DF['mol_id'] = list(smiles_to_molID.values())
    smiles_to_id_DF['smiles'] = list(smiles_to_molID.keys())

    save_id = osp.join(process_path, "smiles_to_mol_id.csv")
    if not osp.exists(save_id):
        smiles_to_id_DF.to_csv(save_id)
        
    Nmols = len(unique_smiles)
    print(
        f"Dataset consists of {len(global_info)} conformations for {Nmols} distinct molecules"
    )

    # test samples are fixed by GeoMol
    test_smiles = pd.read_csv(f"{DATA_PATH}/{dataset}/test_smiles_corrected.csv")
    test_smiles.smiles = test_smiles.smiles.astype(str)
    print(
        f"Selected test set from GeoMol consists of {len(test_smiles)} distinct molecules"
    )

    # create validation samples
    # splits0.npy provided by GeoMol: https://github.com/PattanaikL/GeoMol/tree/main/data
    split_ids = np.load(f"{DATA_PATH}/{dataset}/splits/split0.npy", allow_pickle=True)
    _, val_ids = split_ids[0], split_ids[1]
    val_smiles = [id_to_smi[i] for i in val_ids]
    val_smiles_df = pd.DataFrame()
    val_smiles_df["smiles"] = val_smiles
    val_smiles_df.smiles = val_smiles_df.smiles.astype(str)

    print("Creating test dataframe information")
    test_info = pd.merge(
        left=test_smiles,
        right=global_info,
        how="inner",
        left_on="smiles",
        right_on="smiles",
        left_index=False,
        right_index=False,
        sort=True,
        suffixes=("_x", "_y"),
        copy=True,
        indicator=False,
        validate=None,
    )
    print(
        f"Test dataset consists of {len(test_info)} conformations for {len(set(test_info.smiles))} distinct molecules"
    )

    print("Creating validation dataframe information")
    val_info = pd.merge(
        left=val_smiles_df,
        right=global_info,
        how="inner",
        left_on="smiles",
        right_on="smiles",
        left_index=False,
        right_index=False,
        sort=True,
        suffixes=("_x", "_y"),
        copy=True,
        indicator=False,
        validate=None,
    )
    print(
        f"Validation dataset consists of {len(val_info)} conformations for {len(set(val_info.smiles))} distinct molecules"
    )

    save_test = osp.join(process_path, "test_info.csv")
    save_val = osp.join(process_path, "val_info.csv")
    
    # map the mol-id for smiles
    test_info["mol_id"] = test_info["smiles"].apply(lambda x: smiles_to_molID.get(x))
    val_info["mol_id"] = val_info["smiles"].apply(lambda x: smiles_to_molID.get(x))

    not_identified = test_info["mol_id"].isnull().sum() + val_info["mol_id"].isnull().sum()
    if  not_identified != 0:
        print(f"{not_identified} mol ids are not retrieved based on smiles identifiers for test and val set")

    test_info = test_info[["global_id", "geom_id", "mol_id"]]
    val_info = val_info[["global_id", "geom_id", "mol_id"]]

    if not os.path.exists(save_test) and not os.path.exists(save_val):
        print("Saving test and validation info dataframe")
        test_info.to_csv(save_test)
        val_info.to_csv(save_val)

    # training set
    val_test_global_ids = np.array(
        sorted(list(set(pd.concat([val_info, test_info], axis=0)["global_id"].values)))
    )
    train_global_ids = np.arange(len(global_info))
    train_global_ids = np.setdiff1d(
        train_global_ids, val_test_global_ids, assume_unique=True
    )

    assert len(train_global_ids) + len(val_test_global_ids) == len(global_info)

    train_info = global_info.loc[train_global_ids]
    train_info["mol_id"] = train_info["smiles"].apply(lambda x: smiles_to_molID.get(x))
    not_identified = train_info["mol_id"].isnull().sum()
    if  not_identified != 0:
        print(f"{not_identified} mol ids are not retrieved based on smiles identifiers for training set")

    train_info = train_info[["global_id", "geom_id", "mol_id"]]
    save_train = osp.join(process_path, "train_info.csv")
    if not os.path.exists(save_train):
        print("Saving train info dataframe")
        train_info.to_csv(save_train)

    return None


class GeomDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int = 256,
        num_workers: int = 4,
        transform=MolFeaturization(order=3),
        dataset: str = "drugs",
        env_in_init: bool = False,
        shuffle_train: bool = False,
        # subset_frac: float = 0.1, # old
        max_num_conformers: int = 30,   # -1 means all
        pin_memory: bool = True,
        persistent_workers: bool = True,
    ):
        super().__init__()
        assert dataset in ["qm9", "drugs"]
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        # self.path = osp.join(DATA_PATH, f"{dataset}", "database")   # old, on the gpfs
        self.path = osp.join(DB_READ_PATH, f"{dataset}", "database")   # new, on the hpfs
        self.env_in_init = env_in_init
        self.shuffle_train = shuffle_train
        # self.subset_frac = subset_frac
        self.max_num_conformers = max_num_conformers
        self.transform = transform
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers

    def setup(self, stage: Optional[str] = None) -> None:
        database = GeomLMDBDataset(
            data_file=self.path, transform=self.transform, env_in_init=True
        )

        parent_dir = osp.dirname(self.path)
        
        train_info = pd.read_csv(osp.join(parent_dir, "train_info.csv"))
        val_info = pd.read_csv(osp.join(parent_dir, "val_info.csv"))
        self.test_ids = pd.read_csv(osp.join(parent_dir, "test_info.csv"))["global_id"].tolist()

        if self.max_num_conformers != -1:
            # kinda slow processing now...
            # training
            train_ids = []
            for _, subdf in train_info.groupby("mol_id"):
                ids = subdf["global_id"].values[:self.max_num_conformers]
                train_ids.append(ids)
            train_ids = np.concatenate(train_ids)
                
            # validation
            val_ids = []
            for _, subdf in val_info.groupby("mol_id"):
                ids = subdf["global_id"].values[:self.max_num_conformers]
                val_ids.append(ids)
            val_ids = np.concatenate(val_ids)
        
        self.train_ids = train_ids.tolist()
        self.val_ids = val_ids.tolist()
                
        if stage == "fit" or stage is None:
            self.train_dataset = Subset(dataset=database, indices=self.train_ids)
            self.val_dataset = Subset(dataset=database, indices=self.val_ids)
        if stage == "test" or stage is None:
            self.test_dataset = Subset(dataset=database, indices=self.test_ids)
            
            
    def _setup(self, stage: Optional[str] = None) -> None:
        # old with fractions
        database = GeomLMDBDataset(
            data_file=self.path, transform=self.transform, env_in_init=True
        )

        parent_dir = osp.dirname(self.path)
        
        self.train_ids = pd.read_csv(osp.join(parent_dir, "train_info.csv"))[
            "global_id"
        ].tolist()
        self.val_ids = pd.read_csv(osp.join(parent_dir, "val_info.csv"))[
            "global_id"
        ].tolist()
        self.test_ids = pd.read_csv(osp.join(parent_dir, "test_info.csv"))[
            "global_id"
        ].tolist()

        if self.subset_frac < 1.0:
            Ntrain = len(self.train_ids)
            Nval = len(self.val_ids)
            self.train_ids = self.train_ids[: int(Ntrain * self.subset_frac)]
            self.val_ids = self.val_ids[: int(Nval * self.subset_frac)]
        if stage == "fit" or stage is None:
            self.train_dataset = Subset(dataset=database, indices=self.train_ids)
            self.val_dataset = Subset(dataset=database, indices=self.val_ids)
        if stage == "test" or stage is None:
            self.test_dataset = Subset(dataset=database, indices=self.test_ids)

    def train_dataloader(self, shuffle=False):
        dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=self.shuffle_train,
            persistent_workers=self.persistent_workers,
        )
        return dataloader

    def val_dataloader(self, shuffle=False):
        dataloader = DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=shuffle,
            persistent_workers=self.persistent_workers,
        )
        return dataloader

    def test_dataloader(self, shuffle=False):
        dataloader = DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=shuffle,
            persistent_workers=self.persistent_workers,
        )
        return dataloader


@click.command(
    help="Preprocessing the QM9 or DRUGS conformation datasets from the GEOM Benchmark"
)
@click.option("--dataset", "-d", default="drugs")
@click.option("--max_conformers", "-mc", default=1000)
@click.option("--processes", "-p", default=16)
@click.option("--chunk_size", "-cs", default=2048)
@click.option("--subchunk", "-sc", default=256)
def main(dataset, max_conformers, processes, chunk_size, subchunk):
    print(f"Processing {dataset} Conformation Dataset")

    process_files(
        dataset=dataset,
        max_conformers=max_conformers,
        processes=processes,
        chunk_size=chunk_size,
        subchunk=subchunk,
    )

    print("Processing training/validation and test")
    create_splits(dataset=dataset)


if __name__ == "__main__":
    main()
    