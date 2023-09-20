import pickle
from pathlib import Path


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


CHUNK_SIZE = 5000
SAVE_DIR = Path("/scratch1/seumej/geom_qm/raw")

for set_name in ["test", "train", "val"]:
    pickle_file = Path(f"/scratch1/cremej01/data/geom/raw/{set_name}_data.pickle")
    print(f"Processing {pickle_file}")

    with open(pickle_file, "rb") as f:
        data = pickle.load(f)

    for i, chunk in enumerate(chunks(data, CHUNK_SIZE)):
        chunk_file = pickle_file.stem + f"_{i:04d}" + pickle_file.suffix
        with open(SAVE_DIR / chunk_file, "wb") as f:
            pickle.dump(chunk, f)
