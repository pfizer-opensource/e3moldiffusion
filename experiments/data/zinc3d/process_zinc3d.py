import glob
import gzip
import pickle
from multiprocessing import Pool

import lmdb
import numpy as np
import torch
from experiments.data.ligand.ligand_dataset_nonadaptive import full_atom_encoder
from experiments.data.utils import mol_to_torch_geometric
from rdkit import Chem
from tqdm.auto import tqdm


def parse_sdf_to_dict(rdmol):
    try:
        data = mol_to_torch_geometric(
            rdmol,
            full_atom_encoder,
            smiles=None,
            remove_hydrogens=True,
            cog_proj=True,
            add_ad=True,
            add_pocket=False,
        )
        data = pickle.dumps(data)
    except Exception:
        data = None
    return data


if __name__ == "__main__":
    processed_path = (
        "/hpfs/projects/mlcs/mlhub/e3moldiffusion/data/zinc3d/raw/zinc_dataset.lmdb"
    )
    directory_to_search = "/hpfs/projects/mlcs/mlhub/e3moldiffusion/data/zinc3d"
    pattern = "/**/*.sdf.gz"
    sdf_list = []
    for file_path in glob.iglob(directory_to_search + pattern, recursive=True):
        sdf_list.append(file_path)

    db = lmdb.open(
        processed_path,
        map_size=200 * (1024 * 1024 * 1024),  # 200GB
        create=True,
        subdir=False,
        readonly=False,  # Writable
    )
    index = 0
    index_list = []
    for ix, sdf_supplier in enumerate(tqdm(sdf_list)):
        mol_list = list(
            Chem.ForwardSDMolSupplier(gzip.open(sdf_supplier), removeHs=True)
        )
        torch.multiprocessing.set_sharing_strategy("file_system")
        pool = Pool(processes=16)
        List = pool.map(parse_sdf_to_dict, mol_list)
        pool.close()
        pool.join()
        with db.begin(write=True, buffers=True) as txn:
            for data in List:
                if data is None:
                    continue
                key = str(index).encode()
                txn.put(key=key, value=data)
                index_list.append(key)
                index += 1
    db.close()
    index_list = np.array(index_list)
    np.save(processed_path.split(".")[0] + "_Keys", index_list)
