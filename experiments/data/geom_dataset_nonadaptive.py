from typing import Optional

import numpy as np
import torch
from experiments.data.utils import (Statistics, fully_connected_edge_idx, load_pickle, save_pickle)
from experiments.data.metrics import compute_all_statistics
from pytorch_lightning import LightningDataModule
from rdkit import Chem, RDLogger
from torch_geometric.data import Data, DataLoader, InMemoryDataset
from torch_geometric.utils import subgraph
from tqdm import tqdm


def mol_to_torch_geometric(mol, atom_encoder, smiles):
    adj = torch.from_numpy(Chem.rdmolops.GetAdjacencyMatrix(mol, useBO=True))
    edge_index = adj.nonzero().contiguous().T
    bond_types = adj[edge_index[0], edge_index[1]]
    bond_types[bond_types == 1.5] = 4
    edge_attr = bond_types.long()

    pos = torch.tensor(mol.GetConformers()[0].GetPositions()).float()
    pos = pos - torch.mean(pos, dim=0, keepdim=True)
    atom_types = []
    all_charges = []
    for atom in mol.GetAtoms():
        atom_types.append(atom_encoder[atom.GetSymbol()])
        all_charges.append(atom.GetFormalCharge())        # TODO: check if implicit Hs should be kept

    atom_types = torch.Tensor(atom_types).long()
    all_charges = torch.Tensor(all_charges).long()

    data = Data(x=atom_types, edge_index=edge_index, edge_attr=edge_attr, pos=pos, charges=all_charges,
                smiles=smiles)
    
    data = fully_connected_edge_idx(data=data, without_self_loop=True)
    
    return data


DATAROOT = '/home/let55/workspace/projects/e3moldiffusion/experiments/geom/data'
DATAROOT = '/sharedhome/let55/projects/e3moldiffusion/experiments/geom/data'
# DATAROOT = '/hpfs/userws/let55/projects/e3moldiffusion/experiments/geom/data'

root_no_h = '/home/let55/workspace/projects/e3moldiffusion/experiments/geom/data_noH'

full_atom_encoder = {'H': 0, 'B': 1, 'C': 2, 'N': 3, 'O': 4, 'F': 5, 'Al': 6, 'Si': 7,
                     'P': 8, 'S': 9, 'Cl': 10, 'As': 11, 'Br': 12, 'I': 13, 'Hg': 14, 'Bi': 15}

class GeomDrugsDataset(InMemoryDataset):
    def __init__(self, root,  split,  transform=None, pre_transform=None, pre_filter=None):
        assert split in ['train', 'val', 'test']
        self.split = split
        self.atom_encoder = full_atom_encoder
      
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.statistics = Statistics(num_nodes=load_pickle(self.processed_paths[1]),
                                     atom_types=torch.from_numpy(np.load(self.processed_paths[2])),
                                     bond_types=torch.from_numpy(np.load(self.processed_paths[3])),
                                     charge_types=torch.from_numpy(np.load(self.processed_paths[4])),
                                     valencies=load_pickle(self.processed_paths[5]),
                                     bond_lengths=None, #load_pickle(self.processed_paths[6]),
                                     bond_angles=None) #torch.from_numpy(np.load(self.processed_paths[7])))
        self.smiles = load_pickle(self.processed_paths[8])

    @property
    def raw_file_names(self):
        if self.split == 'train':
            return ['train_data.pickle']
        elif self.split == 'val':
            return ['val_data.pickle']
        else:
            return ['test_data.pickle']

    @property
    def processed_file_names(self):
        h = 'h'
        if self.split == 'train':
            return [f'train_{h}.pt', f'train_n_{h}.pickle', f'train_atom_types_{h}.npy', f'train_bond_types_{h}.npy',
                    f'train_charges_{h}.npy', f'train_valency_{h}.pickle', f'train_bond_lengths_{h}.pickle',
                    f'train_angles_{h}.npy', 'train_smiles.pickle']
        elif self.split == 'val':
            return [f'val_{h}.pt', f'val_n_{h}.pickle', f'val_atom_types_{h}.npy', f'val_bond_types_{h}.npy',
                    f'val_charges_{h}.npy', f'val_valency_{h}.pickle', f'val_bond_lengths_{h}.pickle',
                    f'val_angles_{h}.npy', 'val_smiles.pickle']
        else:
            return [f'test_{h}.pt', f'test_n_{h}.pickle', f'test_atom_types_{h}.npy', f'test_bond_types_{h}.npy',
                    f'test_charges_{h}.npy', f'test_valency_{h}.pickle', f'test_bond_lengths_{h}.pickle',
                    f'test_angles_{h}.npy', 'test_smiles.pickle']

    def download(self):
        raise ValueError('Download and preprocessing is manual. If the data is already downloaded, '
                         f'check that the paths are correct. Root dir = {self.root} -- raw files {self.raw_paths}')

    def process(self):
        RDLogger.DisableLog('rdApp.*')
        print('Raw datapath: ', self.raw_paths[0])
        all_data = load_pickle(self.raw_paths[0])

        data_list = []
        all_smiles = []
        for data in tqdm(all_data, total=len(all_data)):
            smiles, all_conformers = data
            all_smiles.append(smiles)
            for j, conformer in enumerate(all_conformers):
                if j >= 5:
                    break
                data = mol_to_torch_geometric(conformer, full_atom_encoder, smiles)
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)

        print(f"Saving processed data file at {self.processed_paths[0]}")
        torch.save(self.collate(data_list), self.processed_paths[0])

        statistics = compute_all_statistics(data_list, self.atom_encoder, charges_dic={-2: 0, -1: 1, 0: 2,
                                                                                       1: 3, 2: 4, 3: 5},
                                            bonds=False, angles=False
                                            )
        save_pickle(statistics.num_nodes, self.processed_paths[1])
        np.save(self.processed_paths[2], statistics.atom_types)
        np.save(self.processed_paths[3], statistics.bond_types)
        np.save(self.processed_paths[4], statistics.charge_types)
        save_pickle(statistics.valencies, self.processed_paths[5])
        save_pickle(statistics.bond_lengths, self.processed_paths[6])
        np.save(self.processed_paths[7], statistics.bond_angles)
        save_pickle(set(all_smiles), self.processed_paths[8])

def get_data_no_h(data: Data) -> Data:
    select_mask = (data.x != 0)
    edge_index, edge_attr = subgraph(
        subset=select_mask,
        edge_index=data.edge_index,
        edge_attr=data.edge_attr,
        relabel_nodes=True,
        num_nodes=data.num_nodes,
        return_edge_mask=False
    )
    x = data.x[select_mask]
    pos = data.pos[select_mask]
    charges = data.charges[select_mask]
    data = Data(x=x, pos=pos, edge_index=edge_index, edge_attr=edge_attr, charges=charges, smiles=data.smiles)
    data = fully_connected_edge_idx(data)
    assert sum(data.x == 0) == 0
    return data


class GeomDrugsDataset_NoH(InMemoryDataset):
    def __init__(self, root,  split,  transform=None, pre_transform=None, pre_filter=None):
        assert split in ['train', 'val', 'test']
        self.split = split
        self.atom_encoder = full_atom_encoder
      
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.statistics = Statistics(num_nodes=load_pickle(self.processed_paths[1]),
                                     atom_types=torch.from_numpy(np.load(self.processed_paths[2])),
                                     bond_types=torch.from_numpy(np.load(self.processed_paths[3])),
                                     charge_types=torch.from_numpy(np.load(self.processed_paths[4])),
                                     valencies=load_pickle(self.processed_paths[5]),
                                     bond_lengths=None,
                                     bond_angles=None)
    
    def download(self):
        pass
    
    @property
    def raw_file_names(self):
        if self.split == 'train':
            return ['train_data.pickle']
        elif self.split == 'val':
            return ['val_data.pickle']
        else:
            return ['test_data.pickle']
   
    @property
    def processed_file_names(self):
        h = 'noh'
        if self.split == 'train':
            return [f'train_{h}.pt', f'train_n_{h}.pickle', f'train_atom_types_{h}.npy', f'train_bond_types_{h}.npy',
                    f'train_charges_{h}.npy', f'train_valency_{h}.pickle', f'train_bond_lengths_{h}.pickle',
                    f'train_angles_{h}.npy', 'train_smiles.pickle']
        elif self.split == 'val':
            return [f'val_{h}.pt', f'val_n_{h}.pickle', f'val_atom_types_{h}.npy', f'val_bond_types_{h}.npy',
                    f'val_charges_{h}.npy', f'val_valency_{h}.pickle', f'val_bond_lengths_{h}.pickle',
                    f'val_angles_{h}.npy', 'val_smiles.pickle']
        else:
            return [f'test_{h}.pt', f'test_n_{h}.pickle', f'test_atom_types_{h}.npy', f'test_bond_types_{h}.npy',
                    f'test_charges_{h}.npy', f'test_valency_{h}.pickle', f'test_bond_lengths_{h}.pickle',
                    f'test_angles_{h}.npy', 'test_smiles.pickle']

    def process(self):
        if self.split == "train":
            dataset = GeomDrugsDataset(root='/home/let55/workspace/projects/e3moldiffusion/geom/data', split="train")
        elif self.split == "val":
            dataset = GeomDrugsDataset(root='/home/let55/workspace/projects/e3moldiffusion/geom/data', split="val")
        else:
            dataset = GeomDrugsDataset(root='/home/let55/workspace/projects/e3moldiffusion/geom/data', split="test")
        data_list = []
       
        for data in tqdm(dataset, total=len(dataset)):
            data = get_data_no_h(data)
            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])

        statistics = compute_all_statistics(data_list, self.atom_encoder, charges_dic={-2: 0, -1: 1, 0: 2,
                                                                                       1: 3, 2: 4, 3: 5},
                                           bonds=False, angles=False)
        save_pickle(statistics.num_nodes, self.processed_paths[1])
        np.save(self.processed_paths[2], statistics.atom_types)
        np.save(self.processed_paths[3], statistics.bond_types)
        np.save(self.processed_paths[4], statistics.charge_types)
        save_pickle(statistics.valencies, self.processed_paths[5])
        save_pickle(statistics.bond_lengths, self.processed_paths[6])
        np.save(self.processed_paths[7], statistics.bond_angles)
        torch.save(self.collate(data_list), self.processed_paths[0])
        
        
class GeomDataModule(LightningDataModule):
    def __init__(
        self,
        root: Optional[str] = None,
        batch_size: int = 256,
        num_workers: int = 4,
        shuffle_train: bool = True,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        with_hydrogen: bool = True
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root = DATAROOT if root is None else root
        self.shuffle_train = shuffle_train
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.with_hydrogen = with_hydrogen

    def setup(self, stage: Optional[str] = None) -> None:
        print(self.root)
        if not self.with_hydrogen:
            print("loading without hydrogens")
            train_dataset = GeomDrugsDataset_NoH(root=self.root, split="train")
            val_dataset = GeomDrugsDataset_NoH(root=self.root, split="val")
            test_dataset = GeomDrugsDataset_NoH(root=self.root, split="test")
        else:
            print("loading with hydrogens")
            train_dataset = GeomDrugsDataset(root=self.root, split="train")
            val_dataset = GeomDrugsDataset(root=self.root, split="val")
            test_dataset = GeomDrugsDataset(root=self.root, split="test")
            
        if stage == "fit" or stage is None:
            self.train_dataset = train_dataset
            self.val_dataset = val_dataset
            self.test_dataset = test_dataset

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

if __name__ == '__main__':
    from tqdm import tqdm
    print(DATAROOT)
    dataset = GeomDrugsDataset(root=DATAROOT, split="val")
    print(dataset)
    dataset = GeomDrugsDataset(root=DATAROOT, split="test")
    print(dataset)
    dataset = GeomDrugsDataset(root=DATAROOT, split="train")
    print(dataset)
    print(dataset[0])
    print(dataset[0].edge_attr)
    datamodule = GeomDataModule(root=DATAROOT, batch_size=64, num_workers=4)
    datamodule.setup()
    loader = datamodule.val_dataloader()
    for i, data in tqdm(enumerate(loader), total=len(loader)):
        # data = data.cuda()
        pass
    