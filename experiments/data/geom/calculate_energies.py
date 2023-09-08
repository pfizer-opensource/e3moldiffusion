from tqdm import tqdm
import os
import pickle
import argparse
from experiments.xtb_energy import calculate_xtb_energy
from torch_geometric.data.collate import collate
import torch


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description='Energy calculation')
    parser.add_argument('--dataset', type=str, help='Which dataset')
    parser.add_argument('--split', type=str, help='Which data split train/val/test')
    args = parser.parse_args()
    return args


atom_encoder = {
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
atom_decoder = {v: k for k, v in atom_encoder.items()}


def process(dataset, split):
    if dataset == "drugs":
        from experiments.data.geom.geom_dataset_adaptive import (
            GeomDrugsDataset as DataModule,
        )

        root_path = "/hpfs/userws/cremej01/projects/data/geom"
    elif dataset == "qm9":
        from experiments.data.qm9.qm9_dataset import GeomDrugsDataset as DataModule

        root_path = "/hpfs/userws/cremej01/projects/data/qm9"
    else:
        raise ValueError("Dataset not found")

    remove_hs = False

    dataset = DataModule(split=split, root=root_path, remove_h=remove_hs)

    failed_ids = []
    mols = []
    for i, mol in tqdm(enumerate(dataset)):
        atom_types = [atom_decoder[int(a)] for a in mol.x]
        try:
            e, f = calculate_xtb_energy(mol.pos, atom_types)
        except:
            print(f"Molecule with id {i} failed...")
            failed_ids.append(i)
            continue
        mol.energy = e
        mol.forces_norm = f
        mols.append(mol)

    print(f"Collate the data...")
    data, slices = _collate(mols)

    print(f"Saving the data...")
    torch.save((data, slices), (os.path.join(root_path, f"raw/{split}_data_energy.pt")))

    with open(os.path.join(root_path, f"failed_ids_{split}.pickle"), "wb") as f:
        pickle.dump(failed_ids, f)


def _collate(data_list):
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


if __name__ == "__main__":
    args = get_args()
    process(args.dataset, args.split)
