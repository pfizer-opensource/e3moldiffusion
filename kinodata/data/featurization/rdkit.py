import torch
import torch.nn.functional as F
from kinodata.types import RelationType
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.utils import to_undirected

from .atoms import AtomFeatures
from .bonds import BOND_TYPE_TO_IDX, NUM_BOND_TYPES


def atomic_numbers(mol) -> Tensor:
    z = torch.empty(mol.GetNumAtoms(), dtype=torch.long)
    for i, atom in enumerate(mol.GetAtoms()):
        z[i] = atom.GetAtomicNum()
    return z


def atom_positions(mol) -> Tensor:
    conf = mol.GetConformer()
    pos = conf.GetPositions()
    return torch.from_numpy(pos).to(torch.float)


def set_atoms(mol, data: HeteroData, key: str) -> HeteroData:
    data[key].x = atomic_numbers(mol)
    data[key].add_feats = torch.from_numpy(AtomFeatures.compute(mol)).float()
    data[key].pos = atom_positions(mol)
    return data


def bond_tensors(mol):
    row, col = list(), list()
    bond_type_indices = []
    num_nodes = mol.GetNumAtoms()
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        type = bond.GetBondType()
        row.append(i)
        col.append(j)
        bond_type_indices.append(BOND_TYPE_TO_IDX[type])

    edge_index = torch.tensor([row, col], dtype=torch.long).view(2, -1)
    bond_type_indices = torch.tensor(bond_type_indices)

    # one-hot encode bond type
    edge_attr = F.one_hot(bond_type_indices, NUM_BOND_TYPES)

    edge_index, edge_attr = to_undirected(edge_index, edge_attr, num_nodes)
    return edge_index, edge_attr


def set_bonds(mol, data: HeteroData, key: str) -> HeteroData:
    edge_index, edge_attr = bond_tensors(mol)
    data[key, RelationType.Covalent, key].edge_index = edge_index
    data[key, RelationType.Covalent, key].edge_attr = edge_attr

    return data


def append_atoms_and_bonds(mol, data: HeteroData, key: str) -> HeteroData:
    old_num_nodes = data[key].z.size(0)

    data[key].z = torch.cat((data[key].z, atomic_numbers(mol)), dim=0)
    data[key].x = torch.cat(
        (data[key].x, torch.from_numpy(AtomFeatures.compute(mol)).float()), dim=0
    )
    data[key].pos = torch.cat((data[key].pos, atom_positions(mol)), dim=0)

    edge_index, edge_attr = bond_tensors(mol)
    edge_index = edge_index + old_num_nodes

    edge_store = data[key, RelationType.Covalent, key]

    edge_store.edge_index = torch.cat((edge_store.edge_index, edge_index), dim=1)
    edge_store.edge_attr = torch.cat((edge_store.edge_attr, edge_attr), dim=0)
    return data
