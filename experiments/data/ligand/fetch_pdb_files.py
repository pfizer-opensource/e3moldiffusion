import argparse
import os
from pathlib import Path

from tqdm import tqdm

from experiments.data.ligand.process_pdb import get_pdb_components, write_pdb


def main(args):
    protein_files = list(args.files_dir.glob("[!.]*.sdf"))
    pbar = tqdm(protein_files)

    os.makedirs(args.save_dir, exist_ok=True)

    for file in pbar:
        pdb_name = str(file).split("/")[-1].split("-")[0]

        protein, _ = get_pdb_components(pdb_name)
        _ = write_pdb(args.save_dir, protein, pdb_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--files-dir", type=Path)
    parser.add_argument("--save-dir", type=Path)
    args = parser.parse_args()

    main(args)
