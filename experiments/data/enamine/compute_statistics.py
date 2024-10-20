import os
import numpy as np
import lmdb
import pickle
import gzip
import io
from experiments.data.metrics import compute_all_statistics
from experiments.data.utils import Statistics
import argparse
from tqdm import tqdm
def divide_chunks(l, n): 
    for i in range(0, len(l), n):  
        yield l[i:i + n] 


def get_data(env, index):
    with env.begin(write=False) as txn:
        compressed = txn.get(str(index).encode())
        buf = io.BytesIO(compressed)
        with gzip.GzipFile(fileobj=buf, mode="rb") as f:
            serialized = f.read()
        try:
            item = pickle.loads(serialized)
        except Exception as e:
            return None
    return item

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
charges_dic={-2: 0, -1: 1, 0: 2, 1: 3, 2: 4, 3: 5}


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description='Processing script to compute dataset statistics from enamine')
    parser.add_argument('--idx', type=int, help='index for subchunk. Can be 0 until 108 (ends included)', default=0)
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = get_args()   
    dbpath = "/home/let55/workspace/datasets/enamine/database_noH"
    savedir = "/home/let55/workspace/datasets/enamine/stats/"
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    global_id = 0
    env = lmdb.open(str(dbpath), map_size=int(1e13))
    print(env.stat())
    ids = np.arange(env.stat().get('entries'))
    my_chunks = list(divide_chunks(ids, 1_000_000))
    selected_ids = my_chunks[args.idx]
    print(f"Creating dataset from chunk index {args.idx}")
    dataset = []
    for i in tqdm(selected_ids, total=len(selected_ids)):
        data = get_data(env, i)
        dataset.append(data)
    print("Computing statistics")
    stats = compute_all_statistics(
    dataset,
    atom_encoder,
    charges_dic,
    additional_feats = True,
    include_force_norms = False,
    normalize=False
    )
    statssavedir = os.path.join(savedir, f'stats_{args.idx}.pickle')
    print(f"Finished. Saving at {statssavedir}")
    with open(statssavedir, 'wb') as f:
        pickle.dump(obj=stats, file=f)