from glob import glob
from tqdm import tqdm
from rdkit import Chem, RDLogger
from pathlib import Path
import pandas as pd
import argparse

from experiments.sampling.utils import (calculate_sa, calculate_logp, calculate_hdonors, 
                                        calculate_hacceptors, calculate_molwt, calculate_qed,
                                        calculate_lipinski, num_rings, num_aromatic_rings
                                        )


DATA_PATH = "/home/let55/workspace/datasets/enamine/out/"
SAVE_PATH = "/home/let55/workspace/datasets/enamine/"
CHUNK_SDF_FILES = sorted(glob(DATA_PATH + "/chunk_*/out.sdf"))

def compute_stat(mol):
    try:
        out = {
            "sa": calculate_sa(mol).item(),
            "logp": calculate_logp(mol),
            "hdonor": calculate_hdonors(mol),
            "hacceptor": calculate_hacceptors(mol),
            "molwt": calculate_molwt(mol),
            "qed": calculate_qed(mol),
            "lipinski": calculate_lipinski(mol),
            "num_rings": num_rings(mol),
            "num_aromatic_rings": num_aromatic_rings(mol),
            "num_atoms": mol.GetNumAtoms(),
        }
    except: 
        out = None
    return out

def process_chunk(
    chunk_dir, removeHs=True
):
    """
    :param chunk_dir:
    :param processes:
    :param chunk_size:
    :return:
    """

    h = "noh" if removeHs else "h"
    print(f"Process without hydrogens: {removeHs}")
    print(f"Loading chunk sdf file {chunk_dir}")
    mols = [mol for mol in Chem.ForwardSDMolSupplier(chunk_dir, removeHs=removeHs)]
    print(f"Total number of molecules {len(mols)}.")
    
    stats_list = []
    for i, mol in tqdm(enumerate(mols), total=len(mols)):
        out = compute_stat(mol)
        if out is not None:
            stats_list.append(out)
    
    out_df = pd.DataFrame(stats_list)    
    # save
    save_path = str(Path(chunk_dir).parent.absolute()) + f"/rdkit_stats_{h}_out.pickle"
    out_df.to_pickle(save_path)
    print(f"Saved processed file at {save_path}")
    print("Finished!")
    
def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description='Processing script to compute rdkit stats from SDF chunk')
    parser.add_argument('--idx', type=int, help='index for subchunk', default=0)
    parser.add_argument('--removeHs', help='Remove hydrogen from molecules. Defaults to False', default=True, action="store_true")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
   args = get_args()
   process_chunk(chunk_dir=CHUNK_SDF_FILES[args.idx], 
                 removeHs=args.removeHs)