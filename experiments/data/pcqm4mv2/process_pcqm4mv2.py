from tqdm import tqdm
import experiments.data.utils as dataset_utils

from rdkit import Chem
import random
import pickle
from collections import defaultdict
import numpy as np
from glob import glob
from experiments.data.utils import (
    get_rdkit_mol,
    write_xyz_file,
)
import tempfile
from rdkit import Chem
import ase


def read_xyz_file(file_path):
    atom_types = np.genfromtxt(file_path, skip_header=1, usecols=range(1), dtype=str)
    atom_types = np.array([ase.Atom(sym).number for sym in atom_types])
    atom_positions = np.genfromtxt(
        file_path, skip_header=1, usecols=range(1, 4), dtype=np.float32
    )
    return atom_types, atom_positions


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

full_atom_decoder = {v: k for k, v in full_atom_encoder.items()}

FILE_PATH = "/scratch1/cremej01/data/pcqm4mv2/pcqm4m-v2-train.sdf"


def process():
    data_list = []
    suppl = Chem.SDMolSupplier(FILE_PATH)
    for mol in tqdm(suppl):
        try:
            smiles = Chem.MolToSmiles(mol)
            data = dataset_utils.mol_to_torch_geometric(mol, full_atom_encoder, smiles)
            if data.pos.shape[0] != data.x.shape[0]:
                continue
            if data.pos.ndim != 2:
                continue
            if len(data.pos) < 2:
                continue
        except:
            continue
        # even when calling Chem.RemoveHs, hydrogens might be present
        data_list.append(data)

    n_train = int(0.8 * len(data_list))
    n_val = int(0.1 * len(data_list))
    random.shuffle(data_list)

    train_data = data_list[:n_train]
    val_data = data_list[n_train : n_train + n_val]
    test_data = data_list[n_train + n_val :]

    print(f"Number of datapoints: {len(data_list)}.")
    print(f"Number of training samples: {len(train_data)}.")
    print(f"Number of validation samples: {len(val_data)}.")
    print(f"Number of test samples: {len(test_data)}.")

    with open("/scratch1/cremej01/data/pcqm4mv2/raw/train_data.pickle", "wb") as f:
        pickle.dump(train_data, f)

    with open("/scratch1/cremej01/data/pcqm4mv2/raw/val_data.pickle", "wb") as f:
        pickle.dump(val_data, f)

    with open("/scratch1/cremej01/data/pcqm4mv2/raw/test_data.pickle", "wb") as f:
        pickle.dump(test_data, f)


def data_info():
    data_list = []
    errors = 0
    atom_types = defaultdict(int)
    suppl = Chem.SDMolSupplier(FILE_PATH)
    for mol in tqdm(suppl):
        try:
            smiles = Chem.MolToSmiles(mol)
            data = dataset_utils.mol_to_torch_geometric(mol, full_atom_encoder, smiles)
            atoms = [full_atom_decoder[int(a)] for a in data.x]
            for atom in atoms:
                atom_types[atom] += 1

            if data.pos.shape[0] != data.x.shape[0]:
                errors += 1
                continue
            if data.pos.ndim != 2:
                errors += 1
                continue
            if len(data.pos) < 2:
                errors += 1
                continue
        except:
            errors += 1
            continue
        # even when calling Chem.RemoveHs, hydrogens might be present
        data_list.append(data)

    print(f"Number of datapoints: {len(data_list)}.")
    print(f"Number of errors: {errors}.")

    print(f"Atom types: {atom_types}")


def process_xyz():
    raw_paths = ["/scratch1/cremej01/data/pcqm4mv2/pcqm4m-v2_xyz"]
    data_list = []
    errors = 0

    xyz_files = glob(raw_paths[0] + "/*/*.xyz")
    # xyz_files = sorted(xyz_files, key=get_id)

    for xyz in tqdm(xyz_files, desc="Molecules"):
        atoms, coords = read_xyz_file(xyz)

        with tempfile.NamedTemporaryFile() as tmp:
            tmp_file = tmp.name
            # Write xyz file
            write_xyz_file(coords=coords, atom_types=atoms, filename=tmp_file)
            rdkit_mol = get_rdkit_mol(tmp_file)
            smiles = Chem.MolToSmiles(rdkit_mol, isomericSmiles=False)

        try:
            data = dataset_utils.mol_to_torch_geometric(
                rdkit_mol, full_atom_encoder, smiles
            )
        except:
            errors += 1
            continue
        data.mol = rdkit_mol
        data_list.append(data)

    n_train = int(0.8 * len(data_list))
    n_val = int(0.1 * len(data_list))

    random.shuffle(data_list)
    train_data = data_list[:n_train]
    val_data = data_list[n_train : n_train + n_val]
    test_data = data_list[n_train + n_val :]

    print(f"Number of datapoints: {len(data_list)}.")
    print(f"Number of training samples: {len(train_data)}.")
    print(f"Number of validation samples: {len(val_data)}.")
    print(f"Number of test samples: {len(test_data)}.")
    print(f"Number of errors: {len(errors)}.")

    with open("/scratch1/cremej01/data/pcqm4mv2/raw/train_data.pickle", "wb") as f:
        pickle.dump(train_data, f)

    with open("/scratch1/cremej01/data/pcqm4mv2/raw/val_data.pickle", "wb") as f:
        pickle.dump(val_data, f)

    with open("/scratch1/cremej01/data/pcqm4mv2/raw/test_data.pickle", "wb") as f:
        pickle.dump(test_data, f)


if __name__ == "__main__":
    process_xyz()
    # data_info()
