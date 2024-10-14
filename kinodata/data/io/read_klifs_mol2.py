from pathlib import Path
from typing import Union
import pandas as pd
import numpy as np
import re

try:
    from biopandas.mol2 import PandasMol2

    pmol = PandasMol2()
except:
    print("i am so tired of rebuilding my entire docker image every single time")

klifs_mol2_columns = {
    0: ("atom.id", "int32"),
    1: ("atom.name", "string"),
    2: ("atom.x", "float32"),
    3: ("atom.y", "float32"),
    4: ("atom.z", "float32"),
    5: ("atom.type", "string"),
    6: ("residue.subst_id", "int32"),
    7: ("residue.subst_name", "string"),
    8: ("atom.charge", "float32"),
    9: ("atom.status_bit", "string"),
}


def read_klifs_mol2(path: Union[str, Path], with_bonds: bool = True) -> pd.DataFrame:
    path = Path(path)
    atom_df = pmol.read_mol2(str(path), columns=klifs_mol2_columns).df
    if with_bonds:
        raw = path.read_text()
        bond_start = raw.find("@<TRIPOS>BOND")
        bond_len = raw[bond_start + 1 :].find("@")
        rows = raw[bond_start : bond_start + bond_len].split("\n")[1:]
        rows = [row.rstrip().split()[:4] for row in rows]
        bond_df = pd.DataFrame(
            rows, columns=["bond_id", "source_atom_id", "target_atom_id", "bond_type"]
        )
        bond_df["bond_id"] = bond_df["bond_id"].astype(int)
        bond_df["source_atom_id"] = bond_df["source_atom_id"].astype(int)
        bond_df["target_atom_id"] = bond_df["target_atom_id"].astype(int)
        return atom_df, bond_df
    return atom_df
