import gzip
import io
import multiprocessing as mp
import os
import pickle
from glob import glob

import lmdb
import torch
from rdkit import RDLogger
from torch_geometric.data.separate import separate
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def process_files(
    processes: int = 32,
    chunk_size: int = 64,
    subchunk: int = 16,
):
    """
    :param dataset:
    :param max_conformers:
    :param processes:
    :param chunk_size:
    :param subchunk:
    :return:
    """

    save_path = "/hpfs/userws/cremej01/projects/data/zinc3d/raw/zinc_dataset"
    directory_to_search = "/hpfs/scratch/users/cremej01/zinc3d"
    pattern = "*.pt"
    data_list = glob(os.path.join(directory_to_search, pattern))[:100]

    if os.path.exists(save_path):
        print("FYI: Output directory has been created already.")
    chunked_list = list(chunks(data_list, chunk_size))
    chunked_list = [list(chunks(l, subchunk)) for l in chunked_list]

    print(f"Total number of files {len(data_list)}.")
    print(f"Processing {len(chunked_list)} chunks each of size {chunk_size}.")

    env = lmdb.open(str(save_path), map_size=int(1e13))
    global_id = 0
    with env.begin(write=True) as txn:
        for chunklist in tqdm(chunked_list, total=len(chunked_list), desc="Chunks"):
            chunkresult = []
            for datachunk in tqdm(chunklist, total=len(chunklist), desc="Datachunks"):
                with mp.Pool(processes=processes) as pool:
                    res = pool.starmap(func=db_sample_helper, iterable=zip(datachunk))
                    res = [r for r in res if r is not None]
                chunkresult.append(res)

            confs_sub = []
            for cr in chunkresult:
                subconfs = [a["confs"] for a in cr]
                subconfs = [item for sublist in subconfs for item in sublist]
                confs_sub.append(subconfs)

            confs_sub = [item for sublist in confs_sub for item in sublist]

            # save
            for conf in confs_sub:
                result = txn.put(str(global_id).encode(), conf, overwrite=False)
                if not result:
                    raise RuntimeError(
                        f"LMDB entry {global_id} in {str(save_path)} " "already exists"
                    )
                global_id += 1

        print(f"{global_id} molecules have been processed!")
        print("Finished!")


def db_sample_helper(file):
    saved_confs_list = []

    data_list, slices = torch.load(file)
    length = len(data_list.mol)
    for idx in range(length):
        data = separate(
            cls=data_list.__class__,
            batch=data_list,
            idx=idx,
            slice_dict=slices,
            decrement=False,
        )
        # create binary object to be saved
        buf = io.BytesIO()
        with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=6) as f:
            f.write(pickle.dumps(data))
        compressed = buf.getvalue()
        saved_confs_list.append(compressed)
    return {
        "confs": saved_confs_list,
    }


if __name__ == "__main__":
    process_files()
