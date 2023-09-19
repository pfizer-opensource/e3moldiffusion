from tqdm import tqdm
import os
import argparse
import numpy as np
from torch.utils.data import Subset
from tqdm import tqdm
import numpy as np
from experiments.data.utils import save_pickle
from experiments.data.metrics import compute_all_statistics
from torch.utils.data import Subset


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description='Energy calculation')
    parser.add_argument('--dataset', type=str, help='Which dataset')
    parser.add_argument('--split', type=str, help='Which data split train/val/test')
    parser.add_argument('--idx', type=int, default=None, help='Which part of the dataset (pubchem only)')

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

h = "h"
processed_paths = [
    f"train_n_{h}.pickle",
    f"train_atom_types_{h}.npy",
    f"train_bond_types_{h}.npy",
    f"train_charges_{h}.npy",
    f"train_valency_{h}.pickle",
    f"train_bond_lengths_{h}.pickle",
    f"train_angles_{h}.npy",
    "train_smiles.pickle",
]


def process(dataset, split, idx):
    if dataset == "drugs":
        from experiments.data.geom.geom_dataset_adaptive import (
            GeomDrugsDataset as DataModule,
        )

        root_path = "/hpfs/userws/cremej01/projects/data/geom"
    elif dataset == "qm9":
        from experiments.data.qm9.qm9_dataset import QM9Dataset as DataModule

        root_path = "/hpfs/userws/cremej01/projects/data/qm9"
    elif dataset == "aqm":
        from experiments.data.aqm.aqm_dataset_nonadaptive import (
            AQMDataset as DataModule,
        )

        root_path = "/hpfs/userws/cremej01/projects/data/aqm"
    elif dataset == "pubchem":
        from experiments.data.pubchem.pubchem_dataset_nonadaptive import (
            PubChemLMDBDataset as DataModule,
        )

        root_path = "/hpfs/userws/cremej01/projects/data/pubchem/database"
    else:
        raise ValueError("Dataset not found")

    remove_hs = False

    datamodule = DataModule(split=split, root=root_path, remove_h=remove_hs)

    data_list = []
    all_smiles = []

    if dataset == "pubchem":
        split_len = len(datamodule) // 500
        rng = np.arange(0, len(datamodule))
        rng = rng[idx * split_len : (idx + 1) * split_len]
        datamodule = Subset(datamodule, rng)

    for mol in tqdm(datamodule, total=len(datamodule)):
        data_list.append(mol)

    if dataset == "pubchem":
        processed_path = [
            os.path.join(root_path, f"statistics/{idx}_" + path)
            for path in processed_paths
        ]
    else:
        processed_path = [
            os.path.join(root_path, "statistics/" + path) for path in processed_paths
        ]
    statistics = compute_all_statistics(
        data_list,
        atom_encoder,
        charges_dic={-2: 0, -1: 1, 0: 2, 1: 3, 2: 4, 3: 5},
        additional_feats=False,
        normalize=False,
        # do not compute bond distance and bond angle statistics due to time and we do not use it anyways currently
    )
    save_pickle(statistics.num_nodes, processed_path[0])
    np.save(processed_path[1], statistics.atom_types)
    np.save(processed_path[2], statistics.bond_types)
    np.save(processed_path[3], statistics.charge_types)
    save_pickle(statistics.valencies, processed_path[4])
    save_pickle(statistics.bond_lengths, processed_path[5])
    np.save(processed_path[6], statistics.bond_angles)
    save_pickle(set(all_smiles), processed_path[7])


if __name__ == "__main__":
    args = get_args()
    process(args.dataset, args.split, args.idx)
