from glob import glob
from tqdm import tqdm
from rdkit import Chem, RDLogger
import experiments.data.utils as dataset_utils
from pathlib import Path
import argparse
import torch

RDLogger.DisableLog("rdApp.*")


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


DATA_PATH = "/home/let55/workspace/datasets/enamine/out/"
SAVE_PATH = "/home/let55/workspace/datasets/enamine/"
CHUNK_SDF_FILES = sorted(glob(DATA_PATH + "/chunk_*/out.sdf"))

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


def process_chunk(
    chunk_dir, removeHs=False
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

    
   
    all_graphs = []
    all_smiles = []
    for i, mol in tqdm(enumerate(mols), total=len(mols)):
        try:
            smiles = Chem.MolToSmiles(mol)
            data = dataset_utils.mol_to_torch_geometric(
                mol, FULL_ATOM_ENCODER, smiles, remove_hydrogens=removeHs
            )
            if data.pos.shape[0] != data.x.shape[0]:
                continue
            if data.pos.ndim != 2:
                continue
            if len(data.pos) < 2:
                continue
            
            all_graphs.append(data)
            all_smiles.append(smiles)
        except Exception as e:
            print(f"Error index {i}")
            print(e)
            continue
        
    # save
    out_dict = {"graphs": all_graphs, "smiles": all_smiles}
    save_path = str(Path(chunk_dir).parent.absolute()) + f"/graphs_{h}_out.pt"
    torch.save(obj=out_dict, f=save_path)
    print(f"Saved processed file at {save_path}")
    print("Finished!")

def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description='Processing script to create pytorch geometric graphs from SDF chunk')
    parser.add_argument('--idx', type=int, help='index for subchunk', default=0)
    parser.add_argument('--removeHs', help='Remove hydrogen from molecules. Defaults to False', default=False, action="store_true")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
   args = get_args()
   process_chunk(chunk_dir=CHUNK_SDF_FILES[args.idx], 
                 removeHs=args.removeHs)