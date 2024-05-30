import gzip
import io
import os
import pickle
from glob import glob
from multiprocessing import Pool

import lmdb
import torch
from tqdm import tqdm


def compress_object_to_binary(data):
    # create binary object to be saved
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=6) as f:
        f.write(pickle.dumps(data))
    compressed = buf.getvalue()
    return compressed


def get_data(db_path):
    db = lmdb.open(
        str(db_path),
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        create=False,
    )
    with db.begin() as _txn:
        keys = list(_txn.cursor().iternext(values=False))
        data = [
            compress_object_to_binary(pickle.loads(_txn.get(key))) for key in tqdm(keys)
        ]
    return data


if __name__ == "__main__":
    save_path = "/hpfs/userws/cremej01/projects/data/zinc3d/raw/zinc_dataset"
    directory_to_search = "/hpfs/scratch/users/cremej01/zinc3d"
    pattern = "*"
    db_paths = glob(os.path.join(directory_to_search, pattern))

    env = lmdb.open(str(save_path), map_size=int(1e13))
    print(env.stat())
    global_id = 0
    with env.begin(write=True) as txn:
        torch.multiprocessing.set_sharing_strategy("file_system")
        pool = Pool(processes=32)
        List = pool.map(get_data, db_paths)
        pool.close()
        pool.join()
        for data in tqdm(List):
            for d in data:
                try:
                    result = txn.put(str(global_id).encode(), d, overwrite=False)
                    global_id += 1
                except Exception as e:
                    print(f"Error at file {d}")
                    print(e)
                    continue
