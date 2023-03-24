import torch
from tqdm import tqdm
import torch as pt
from glob import glob
from torch_geometric.data import Data, Dataset
import pandas as pd
import numpy as np
import os
from typing import Optional, Union
import numpy as np
import pandas as pd
import torch
from e3moldiffusion.molfeat import get_bond_feature_dims, smiles_or_mol_to_graph
from torch_geometric.data import Data
from tqdm import tqdm
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
from openbabel import pybel
from torch_geometric.data.collate import collate
from torch_geometric.data.separate import separate
from rdkit import Chem
from openbabel import pybel


def atom_type_config(dataset: str = "qm9"):
    if dataset == "qm9":
        mapping = {"H": 0, "C": 1, "N": 2, "O": 3, "F": 4}
    elif dataset == "fullerene":
        mapping = {"H": 0, "C": 1, "N": 2, "Cl": 3}
    elif dataset == "drugs":
        mapping =  {
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
    else:
        raise ValueError("Dataset not found!")
    return mapping

class MolFeaturization:
    def __init__(self, dataset: str = "fullerene",
                 create_bond_graph: bool = True,
                 save_smiles: bool = True,
                 fully_connected_edge_index=True):
        super().__init__()
        assert dataset in ["fullerene", "qm9", "drugs"]  
        self.create_bond_graph = create_bond_graph
        self.save_smiles = save_smiles
        self.get_fully_connected_edge_idx = fully_connected_edge_index
        self.mapping = atom_type_config(dataset=dataset)
        
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
        data = smiles_or_mol_to_graph(smol=mol, create_bond_graph=self.create_bond_graph)
        # not error-prone as of now...
        data.z_norm = torch.tensor([self.mapping.get(el) for el in data.atom_elements])

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
    
    def __call__(self, mol) -> Data:
        data = self.featurize_smiles_or_mol(smiles_mol=mol)
        if self.get_fully_connected_edge_idx:
            data = self.fully_connected_edge_idx(data=data, without_self_loop=True)
        return data
    
    
def get_rdkit_mol(fname_xyz):
    mol = next(pybel.readfile("xyz", fname_xyz))
    mol = Chem.MolFromPDBBlock(molBlock=mol.write(format="pdb"),
                               sanitize=False,
                               removeHs=False,
                               proximityBonding=True
                               )
    assert len(Chem.GetMolFrags(mol)) == 2
    return mol

class Fullerene(Dataset):
    """
    Fullerene dataset
    """
    HARTREE_TO_EV = 27.211386246
    BORH_TO_ANGSTROM = 0.529177
    
    element_numbers = {"H": 1, "C": 6, "N": 7, "Cl": 17}

    @property
    def raw_file_names(self):
        return f""
    
    @property
    def processed_file_names(self):
        return [
            f"{self.dataset}.pt",
        ]

    def __init__(
        self,
        root=None,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        dataset='toluene',
    ):

        self.dataset = dataset
        self.featurization = MolFeaturization(
            dataset="fullerene", 
            create_bond_graph=True,
            fully_connected_edge_index=True
            )
        
        if self.dataset == "toluene" or self.dataset == "tca" or self.dataset == "tcb" or self.dataset == "c60":
            self.self_energies = {"H": -0.73745377147553448,
                            "C": -1.79392597221240613,
                            "N": -2.61051408817916508,
                            "Cl": -4.48197270972277195,
                            }
        elif self.dataset == "acetonitrile":
            self.self_energies = {'H': -0.8046451072539278,
                            'C': -1.7950813996858133,
                            'N': -2.6128727708350983,
                            'Cl': -4.481866526663341
                            }

        elif self.dataset == "dichloromethane":
            self.self_energies = {'H': -0.7505127977017374,
                            'C': -1.793603498257669,
                            'N': -2.611837965873026,
                            'Cl': -4.482184518379966
                            }

        elif self.dataset == "chloroform":
            self.self_energies = {'H': -0.758762433068001,
                            'C': -1.794824710190249,
                            'N': -2.6117351238423696,
                            'Cl': -4.482891949208439
                            }

        elif self.dataset == "hexane":
            self.self_energies = {'H': -0.7133216464403819,
                            'C': -1.7937795250469324,
                            'N': -2.607640133714646,
                            'Cl': -4.4821706161229695}

        else:
            raise ValueError("Dataset does not exist!")

        super().__init__(root, transform, pre_transform, pre_filter)
        
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    def process(self):

        assert len(self.raw_paths) == 1        
        path = self.raw_paths[0]

        xyz_files = sorted(glob(os.path.join(path, "*.xyz")))
        energies = list(pd.read_csv(os.path.join(path, "Energies.csv"))['geometry;energy (Eh)'])

        data_list = []
        for i, xyz in tqdm(enumerate(xyz_files)):
            
            idx_xyz = int(''.join(list(filter(str.isdigit, xyz.split("/")[-1]))))
            
            rdkit_mol = get_rdkit_mol(fname_xyz=xyz)       
            idx_e, energy = energies[idx_xyz].split(";")
            idx_e = int(idx_e)
            energy = torch.tensor([float(energy)]).unsqueeze(0)
            data = self.featurization(rdkit_mol)
            data.pos = torch.from_numpy(rdkit_mol.GetConformers()[0].GetPositions()).float()
            data.y = energy
            data_list.append(data)
        data, slices = self._collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def len(self):
        return len(self.data.y)

    def get(self, idx: int) -> Data:

        data = separate(
            cls=self.data.__class__,
            batch=self.data,
            idx=idx,
            slice_dict=self.slices,
            decrement=False,
        )

        return data

    def _collate(self, data_list):
        r"""Collates a Python list of :obj:`torch_geometric.data.Data` objects
        to the internal storage format of
        :class:`~torch_geometric.data.InMemoryDataset`."""
        if len(data_list) == 1:
            return data_list[0], None

        data, slices, _ = collate(
            data_list[0].__class__,
            data_list=data_list,
            increment=False,
            add_batch=False,
        )
        return data, slices

    def get_atomref(self, max_z=100):
        out = pt.zeros(max_z)
        out[list(self.element_numbers.values())] = pt.tensor(
            list(self.self_energies.values())
        )
        return out.view(-1, 1)
    
    
if __name__ == "__main__":
    xyz_files = sorted(glob(os.path.join("/Users/tuanle/Desktop/projects/e3moldiffusion/fullerene/data/Hexane/raw", "*.xyz")))
    featurizer = MolFeaturization(dataset="fullerene",
                                create_bond_graph=True,
                                save_smiles=True,
                                fully_connected_edge_index=True
                                )
    data = featurizer(get_rdkit_mol(fname_xyz=xyz_files[0]))
    dataset = Fullerene(root="/Users/tuanle/Desktop/projects/e3moldiffusion/fullerene/data/Hexane",
                        dataset="hexane")
    print(dataset)