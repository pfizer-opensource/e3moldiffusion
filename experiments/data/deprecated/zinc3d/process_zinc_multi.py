import argparse
import glob
import gzip
import os
import pickle

import lmdb
from experiments.data.utils import mol_to_torch_geometric
from experiments.utils import split_list
from rdkit import Chem, RDLogger
from torch_geometric.data.collate import collate
from torch_geometric.data.separate import separate
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")


FULL_ATOM_ENCODER = {
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


def parse_sdf_to_dict(mol_list):
    data_list = []
    for rdmol in mol_list:
        try:
            data = mol_to_torch_geometric(
                rdmol,
                FULL_ATOM_ENCODER,
                smiles=None,
                remove_hydrogens=True,
                cog_proj=True,
                add_ad=True,
                add_pocket=False,
            )
            data_list.append(data)
        except Exception:
            continue
    return data_list


def process_files(args):
    """
    :param dataset:
    :param max_conformers:
    :param processes:
    :param chunk_size:
    :param subchunk:
    :return:
    """

    pattern = "/**/*.sdf.gz"
    data_list = []
    for file_path in glob.iglob(args.directory_to_search + pattern, recursive=True):
        data_list.append(file_path)
    print(f"Process without hydrogens: {args.remove_hs}")

    if os.path.exists(args.save_path):
        print("FYI: Output directory has been created already.")

    data_list = split_list(data_list, args.num_cpus)[args.mp_index - 1]
    print(f"Processing {len(data_list)} SDF files on job index {args.mp_index}.")

    data = []
    for ix, sdf_supplier in enumerate(tqdm(data_list)):
        mol_list = list(
            Chem.ForwardSDMolSupplier(gzip.open(sdf_supplier), removeHs=True)
        )
        mols = parse_sdf_to_dict(mol_list)
        if len(mols) > 0:
            data.extend(mols)

    if len(data) > 0:
        # data, slices = _collate(data)
        # torch.save(
        #     (data, slices), os.path.join(args.save_path, f"{args.mp_index}_data.pt")
        # )
        global_id = 0
        env = lmdb.open(
            os.path.join(args.save_path, f"{args.mp_index}_data"), map_size=int(10e9)
        )
        print(env.stat())
        with env.begin(write=True) as txn:
            for i, file in enumerate(tqdm(data)):
                try:
                    compressed = pickle.dumps(file)
                    _ = txn.put(str(global_id).encode(), compressed, overwrite=False)
                    global_id += 1
                    print()
                except Exception as e:
                    print(f"Error at file {file}")
                    print(e)
                    continue
    print("Done!")


def _collate(data_list):
    if len(data_list) == 1:
        return data_list[0], None
    data, slices, _ = collate(
        data_list[0].__class__,
        data_list=data_list,
        increment=False,
        add_batch=False,
    )
    return data, slices


def get(data_list, slices, idx):
    data = separate(
        cls=data_list.__class__,
        batch=data_list,
        idx=idx,
        slice_dict=slices,
        decrement=False,
    )
    return data


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description='Data generation')
    parser.add_argument('--mp-index', default=0, type=int)
    parser.add_argument("--save-path", default="/hpfs/projects/mlcs/mlhub/e3moldiffusion/data/zinc3d/raw", type=str)
    parser.add_argument("--directory-to-search", default="/hpfs/projects/mlcs/mlhub/e3moldiffusion/data/zinc3d", type=str)
    parser.add_argument("--num-cpus", default=32, type=int)
    parser.add_argument("--remove-hs", action="store_true")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    process_files(args)
