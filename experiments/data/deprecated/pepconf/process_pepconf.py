import multiprocessing as mp
from tqdm import tqdm
from glob import glob
import os
import numpy as np
import ase
from ase.io import write, read
from ase import Atoms
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from tqdm import tqdm
import multiprocessing as mp
import pickle
from experiments.data.utils import *
import tempfile


def get_mol(xyz):
    charges = [0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6]
    raw_mol = Chem.MolFromXYZFile(xyz)
    mol = Chem.Mol(raw_mol)
    for charge in charges:
        try:
            rdDetermineBonds.DetermineBonds(mol, charge=charge)
        except:
            continue
        else:
            break
    smiles = Chem.MolToSmiles(mol)
    return smiles, mol


def get_mol_babel(xyz):
    atoms = read(xyz)
    with tempfile.NamedTemporaryFile() as tmp:
        tmp_file = tmp.name
        # Write xyz file
        write_xyz_file(
            coords=atoms.positions, atom_types=atoms.numbers, filename=tmp_file
        )
        rdkit_mol = get_rdkit_mol(tmp_file)
        smiles = Chem.MolToSmiles(rdkit_mol, isomericSmiles=False)

    return smiles, rdkit_mol


def split(iter, n):
    k, m = divmod(len(iter), n)
    split_data = [
        iter[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n)
    ]
    split_data_order_number = [[i, v] for i, v in enumerate(split_data)]
    return split_data_order_number


def func(x, q1, q2):
    index = x[0]  # index of sublist
    value = x[1]  # content of each sublist
    res1 = []
    print(f"Job {index} starting\n")
    for i in tqdm(value):
        res1.append(get_mol(i))
    q1.put(res1)
    q2.put(index)
    print(f"Job {index} finishing\n")


if __name__ == "__main__":
    xyz_files = []
    peptides = ["bioactive", "cyclic", "dipeptide", "disulfide", "tripeptide"]
    for peptide in peptides:
        xyz_path = f"/scratch1/cremej01/data/pepconf/{peptide}/xyz"
        xyz_files += glob(os.path.join(xyz_path, "*.xyz"))

    nprocs = mp.cpu_count()
    print(f"Number of CPU cores: {nprocs}")

    sub_list = split(xyz_files, nprocs)
    qout1 = mp.Queue()
    qout2 = mp.Queue()
    processes = [mp.Process(target=func, args=(sub, qout1, qout2)) for sub in sub_list]

    for p in processes:
        p.daemon = True
        p.start()

    unsorted_result = [[qout1.get(), qout2.get()] for p in processes]
    result = sum([t[0] for t in sorted(unsorted_result, key=lambda x: x[1])], [])

    for p in processes:
        p.join()
        p.close()
    print("All task is done!")

    with open("/scratch1/cremej01/data/pepconf/raw/train_data.pickle", "wb") as f:
        pickle.dump(result, f)
