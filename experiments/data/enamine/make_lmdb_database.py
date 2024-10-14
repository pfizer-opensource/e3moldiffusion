from glob import glob
import os
import lmdb
from tqdm import tqdm
import gzip
import torch
import io
import pickle

def compress_object_to_binary(data):
    # create binary object to be saved
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=6) as f:
        f.write(pickle.dumps(data))
    compressed = buf.getvalue()
    return compressed

if __name__ == "__main__":
    files = sorted(glob("/home/let55/workspace/datasets/enamine/out/chunk_*/graphs*.pt"))
    save_path = "/home/let55/workspace/datasets/enamine/database_noH"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    global_id = 0
    env = lmdb.open(str(save_path), map_size=int(1e13))
    print(env.stat())
    with env.begin(write=True) as txn:
        for i, file in enumerate(files):
            print(f"Chunk {i}/{len(files)}")
            try:
                dataset = torch.load(file)["graphs"]
                for data in tqdm(dataset, total=len(dataset)):
                    compressed = compress_object_to_binary(data)
                    result = txn.put(str(global_id).encode(), compressed, overwrite=False)
                    global_id += 1
                print()
            except Exception as e:
                print(f"Error at file {file}")
                print(e)
                continue
            
    print("Finished")