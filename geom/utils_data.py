from torch_geometric.data import Data
from typing import Optional, Union
from e3moldiffusion.molfeat import smiles_or_mol_to_graph
from torch_geometric.data import Data
from rdkit import Chem, RDLogger
from openbabel import pybel
from rdkit import Chem
from openbabel import pybel
from rdkit import Chem
import torch
from torch_geometric.data import Data
from torch_geometric.utils import (
    to_dense_adj,
    to_dense_batch,
    remove_self_loops,
    sort_edge_index
)
from torch_sparse import coalesce
import torch.nn.functional as F
import pickle
import rmsd
import numpy as np

RDLogger.DisableLog("rdApp.*")


def write_xyz_file(coords, atom_types, filename):
    out = f"{len(coords)}\n\n"
    assert len(coords) == len(atom_types)
    for i in range(len(coords)):
        out += f"{atom_types[i]} {coords[i, 0]:.3f} {coords[i, 1]:.3f} {coords[i, 2]:.3f}\n"
    with open(filename, "w") as f:
        f.write(out)


def save_pickle(array, path):
    with open(path, "wb") as f:
        pickle.dump(array, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def get_rdkit_mol(fname_xyz):
    mol = next(pybel.readfile("xyz", fname_xyz))
    mol = Chem.MolFromPDBBlock(
        molBlock=mol.write(format="pdb"),
        sanitize=False,
        removeHs=False,
        proximityBonding=True,
    )
    # assert len(Chem.GetMolFrags(mol)) == 2
    return mol


def calc_rmsd(mol1, mol2):
    U = rmsd.kabsch(mol1, mol2)
    mol1 = np.dot(mol1, U)
    return rmsd.rmsd(mol1, mol2)


def create_bond_graph(data, atom_encoder):
    mol = data.mol
    adj = torch.from_numpy(Chem.rdmolops.GetAdjacencyMatrix(mol, useBO=True))
    edge_index = adj.nonzero().contiguous().T
    bond_types = adj[edge_index[0], edge_index[1]]
    bond_types[bond_types == 1.5] = 4
    edge_attr = bond_types.long()

    pos = torch.tensor(mol.GetConformers()[0].GetPositions()).float()
    assert calc_rmsd(pos.numpy(), data.pos.numpy()) < 1.0e-3

    atom_types = []
    all_charges = []
    for atom in mol.GetAtoms():
        atom_types.append(atom_encoder[atom.GetSymbol()])
        all_charges.append(atom.GetFormalCharge())
    atom_types = torch.Tensor(atom_types).long()
    assert (atom_types == data.x).all()

    all_charges = torch.Tensor(all_charges).long()
    data.charges = all_charges

    # edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
    # E = to_dense_adj(
    #     edge_index=edge_index,
    #     batch=torch.zeros_like(atom_types),
    #     edge_attr=edge_attr,
    #     max_num_nodes=len(atom_types),
    # )
    # diag_mask = ~torch.eye(5, dtype=torch.bool)
    # E = F.one_hot(E, num_classes=5).float() * diag_mask
    data.bond_index = edge_index
    data.bond_attr = edge_attr

    data = fully_connected_edge_idx(data=data, without_self_loop=True)

    return data


def fully_connected_edge_idx(data: Data, without_self_loop: bool = True):
    N = data.pos.size(0)
    row = torch.arange(N, dtype=torch.long)
    col = torch.arange(N, dtype=torch.long)
    row = row.view(-1, 1).repeat(1, N).view(-1)
    col = col.repeat(N)
    fc_edge_index = torch.stack([row, col], dim=0)
    if without_self_loop:
        mask = fc_edge_index[0] != fc_edge_index[1]
        fc_edge_index = fc_edge_index[:, mask]
        
    fc_edge_index = sort_edge_index(fc_edge_index, sort_by_row=False, num_nodes=N)
    data.fc_edge_index = fc_edge_index
    
    return data


def atom_type_config(dataset: str = "qm9"):
    if dataset == "qm9":
        mapping = {"H": 0, "C": 1, "N": 2, "O": 3, "F": 4}
    elif dataset == "aqm":
        mapping = {"H": 0, "C": 1, "N": 2, "O": 3, "F": 4, "P": 5, "S": 6, "Cl": 7}
    elif dataset == "fullerene":
        mapping = {"H": 0, "C": 1, "N": 2, "Cl": 3}
    elif dataset == "drugs":
        mapping = {
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
    def __init__(
        self,
        dataset: str = "fullerene",
        create_bond_graph: bool = True,
        save_smiles: bool = True,
        fully_connected_edge_index=True,
    ):
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
        data = smiles_or_mol_to_graph(
            smol=mol, create_bond_graph=self.create_bond_graph
        )
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
        edge_index = sort_edge_index(edge_index, num_nodes=N, sort_by_row=False)
        data.edge_index_fc = edge_index
        return data

    def __call__(self, mol) -> Data:
        data = self.featurize_smiles_or_mol(smiles_mol=mol)
        if self.get_fully_connected_edge_idx:
            data = self.fully_connected_edge_idx(data=data, without_self_loop=True)
        return data


class Statistics:
    def __init__(
        self,
        num_nodes,
        atom_types,
        bond_types,
        charge_types,
        valencies,
        bond_lengths,
        bond_angles,
    ):
        self.num_nodes = num_nodes
        # print("NUM NODES IN STATISTICS", num_nodes)
        self.atom_types = atom_types
        self.bond_types = bond_types
        self.charge_types = charge_types
        self.valencies = valencies
        self.bond_lengths = bond_lengths
        self.bond_angles = bond_angles
