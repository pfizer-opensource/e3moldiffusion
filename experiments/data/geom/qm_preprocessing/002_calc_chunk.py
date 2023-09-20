import concurrent.futures
import pickle
from pathlib import Path

from experiments.xtb_wrapper import xtb_calculate
from rdkit import Chem
from tqdm import tqdm

NUM_WORKERS = 24
XTB_EXE = "xtb"
XTB_OPTIONS = {}
GLOBAL_SCR = "/scratch1/seumej/tmp"
SAVE_DIR = Path("/scratch1/seumej/geom_qm/qm")


def wrap_rdkit_xtb(mol, confid=0, options=XTB_OPTIONS):
    conf = mol.GetConformer(confid)
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    coords = conf.GetPositions()

    charge = Chem.GetFormalCharge(mol)
    nUnpairedRadials = 0
    for a in mol.GetAtoms():
        numRadicals = a.GetNumRadicalElectrons()
        if numRadicals % 2 == 1:
            nUnpairedRadials += 1
    multiplicity = nUnpairedRadials + 1

    results = xtb_calculate(
        atoms,
        coords,
        charge=charge,
        multiplicity=multiplicity,
        options=options,
        xtb_cmd=XTB_EXE,
        scr=GLOBAL_SCR,
    )
    try:
        assert conf.Is3D()
        assert mol.GetNumAtoms() == Chem.AddHs(mol).GetNumAtoms()
    except AssertionError as e:
        results["error"] = str(e)

    return results


def run_list_of_mols(mols, num_workers):
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = executor.map(wrap_rdkit_xtb, mols)
    return [res for res in results]


def run_chunk(filename):
    filename = Path(filename)

    with open(filename, "rb") as f:
        data = pickle.load(f)

    new_data = []
    for d in tqdm(data):
        _, mols = d
        results = run_list_of_mols(mols, num_workers=NUM_WORKERS)
        d += (results,)
        new_data.append(d)

    outfile = SAVE_DIR / (filename.stem + "_qm" + filename.suffix)
    print(f"Writing output to {outfile}")
    with open(outfile, "wb") as f:
        pickle.dump(new_data, f)


if __name__ == "__main__":
    import sys

    filename = str(sys.argv[-1])
    print(f"Calculating on file {filename}")

    run_chunk(filename)
