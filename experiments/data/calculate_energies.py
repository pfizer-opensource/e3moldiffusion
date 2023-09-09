from tqdm import tqdm
import os
import pickle
import argparse
from experiments.xtb_energy import calculate_xtb_energy
from torch_geometric.data.collate import collate
import torch
import numpy as np
from torch.utils.data import Subset


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description='Energy calculation')
    parser.add_argument('--dataset', type=str, help='Which dataset')
    parser.add_argument('--split', type=str, help='Which data split train/val/test')
    parser.add_argument('--idx', type=int, help='Which part of the dataset (pubchem only)')

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


def process(dataset, split, idx):
    if dataset == "drugs":
        from experiments.data.geom.geom_dataset_adaptive import (
            GeomDrugsDataset as DataModule,
        )

        root_path = "/hpfs/userws/cremej01/projects/data/geom"
    elif dataset == "qm9":
        from experiments.data.qm9.qm9_dataset import GeomDrugsDataset as DataModule

        root_path = "/hpfs/userws/cremej01/projects/data/qm9"
    elif dataset == "pubchem":
        from experiments.data.pubchem.pubchem_dataset_nonadaptive import (
            PubChemLMDBDataset as DataModule,
        )

        root_path = "/hpfs/userws/cremej01/projects/data/pubchem/database"
    else:
        raise ValueError("Dataset not found")

    remove_hs = False

    datamodule = DataModule(split=split, root=root_path, remove_h=remove_hs)

    energies = []

    if dataset == "pubchem":
        split_len = len(datamodule) // 500
        rng = np.arange(0, len(datamodule))
        rng = rng[idx * split_len : (idx + 1) * split_len]
        datamodule = Subset(datamodule, rng)

    for i, mol in tqdm(enumerate(datamodule), total=len(datamodule)):
        atom_types = [atom_decoder[int(a)] for a in mol.x]
        try:
            e, f = calculate_xtb_energy(mol.pos, atom_types)
        except:
            print(f"Molecule with id {i} failed...")
            # failed_ids.append(i)
            continue
        energies.append(e)

    with open(os.path.join(root_path, f"energies_{split}_{idx}.pickle"), "wb") as f:
        pickle.dump(energies, f)

    # with open(os.path.join(root_path, f"failed_ids_{split}_{idx}.pickle"), "wb") as f:
    #     pickle.dump(failed_ids, f)


if __name__ == "__main__":
    args = get_args()
    process(args.dataset, args.split, args.idx)
