import argparse
import os
import pickle
import random
import tempfile

import experiments.data.utils as dataset_utils
import h5py
import numpy as np
import torch
from experiments.data.utils import (
    get_rdkit_mol,
    write_xyz_file,
)
from rdkit import Chem
from tqdm import tqdm

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
mol_properties = [
    "DIP",
    "HLgap",
    "eAT",
    "eC",
    "eEE",
    "eH",
    "eKIN",
    "eKSE",
    "eL",
    "eNE",
    "eNN",
    "eMBD",
    "eTS",
    "eX",
    "eXC",
    "eXX",
    "mPOL",
]


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description='Data generation')
    parser.add_argument('--file-path', default=None, type=str, help='Data file path')
    parser.add_argument('--out-path', default=None, type=str, help='Processed data file path')
    args = parser.parse_args()
    return args


def split_data(file_path, out_path):
    data_list = []
    for mol in tqdm(h5py.File(file_path[0]).values(), desc="Molecules"):
        for group_name in mol:
            group = mol[group_name]

            all_prop = []
            z = torch.tensor(np.array(group["atNUM"]), dtype=torch.long)
            all_pos = torch.tensor(np.array(group["atXYZ"]), dtype=torch.float32)
            # all_dy = torch.tensor(np.array(group["totFOR"]), dtype=torch.float32)
            for prop in mol_properties:
                tmp = torch.tensor(np.array(group[prop]), dtype=torch.float32)
                all_prop.append(tmp)
            y = torch.cat(all_prop).unsqueeze(0)

            # assert all_pos.shape[0] == all_dy.shape[0]
            assert all_pos.shape[0] == z.shape[0]
            assert all_pos.shape[1] == 3

            # assert all_dy.shape[0] == z.shape[0]
            # assert all_dy.shape[1] == 3
            assert y.shape[0] == 1
            assert y.shape[1] == len(mol_properties)

            with tempfile.NamedTemporaryFile() as tmp:
                tmp_file = tmp.name
                # Write xyz file
                write_xyz_file(coords=all_pos, atom_types=z, filename=tmp_file)
                rdkit_mol = get_rdkit_mol(tmp_file)
                smiles = Chem.MolToSmiles(rdkit_mol, isomericSmiles=False)

            data = dataset_utils.mol_to_torch_geometric(
                rdkit_mol,
                full_atom_encoder,
                smiles,
                remove_hydrogens=False,
            )
            data.y = y
            data.mol = rdkit_mol
            data_list.append(data)

    n_train = int(0.8 * len(data_list))
    n_val = int(0.1 * len(data_list))

    random.shuffle(data_list)

    train_data = data_list[: n_train + 2]
    val_data = data_list[n_train + 2 : n_train + 2 + n_val]
    test_data = data_list[n_train + n_val + 2 :]

    os.makedirs(os.path.join(out_path, "raw"), exist_ok=True)

    with open(os.path.join(out_path, "raw/train_data.pickle"), "wb") as f:
        pickle.dump(train_data, f)

    with open(os.path.join(out_path, "raw/val_data.pickle"), "wb") as f:
        pickle.dump(val_data, f)

    with open(os.path.join(out_path, "raw/test_data.pickle"), "wb") as f:
        pickle.dump(test_data, f)


if __name__ == "__main__":
    args = get_args()
    split_data(args.file_path, args.out_path)
