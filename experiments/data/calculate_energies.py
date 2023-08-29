from tqdm import tqdm
import os
import pickle
import argparse
from experiments.xtb_energy import calculate_xtb_energy


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description='Energy calculation')
    parser.add_argument('--dataset', type=str, help='Which dataset')
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


def process(dataset):
    if dataset == "drugs":
        from experiments.data.geom.geom_dataset_adaptive import (
            GeomDrugsDataset as DataModule,
        )

        root_path = "/hpfs/userws/cremej01/data/geom"
    elif dataset == "qm9":
        from experiments.data.qm9.qm9_dataset import GeomDrugsDataset as DataModule

        root_path = "/hpfs/userws/cremej01/data/qm9"

    remove_hs = False

    train_dataset = DataModule(split="train", root=root_path, remove_h=remove_hs)

    energies = []
    forces_norm = []
    for mol in tqdm(train_dataset):
        atom_types = [atom_decoder[int(a)] for a in mol.x]
        try:
            e, f = calculate_xtb_energy(mol.pos, atom_types)
        except:
            continue
        energies.append(e)
        forces_norm.append(f)

    with open(os.path.join(root_path, "energies.pickle"), "wb") as f:
        pickle.dump(energies, f)
    with open(os.path.join(root_path, "forces_norms.pickle"), "wb") as f:
        pickle.dump(forces_norm, f)


if __name__ == "__main__":
    args = get_args()
    process(args.dataset)
