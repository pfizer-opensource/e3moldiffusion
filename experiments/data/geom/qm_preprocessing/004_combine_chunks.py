import pickle
import shutil

from pathlib import Path

qm_path = Path("/scratch1/seumej/geom_qm/qm")
sets = ["test", "train", "val"]


for s in sets:
    print(s)
    names = [f for f in qm_path.glob(f"{s}_*.pickle")]
    names.sort()

    all_data = []
    for f in names:
        with open(str(f), "rb") as readfile:
            data = pickle.load(readfile)
        all_data.extend(data)
    print("now writing")
    with open(qm_path / f"{s}_data_qm.pickle", "wb") as writefile:
        pickle.dump(all_data, writefile)
