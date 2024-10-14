from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from kinodata.types import NodeType

_DATA = Path(__file__).parents[3] / "data"

sitealign_feature_lookup = pd.DataFrame.from_dict(
    {
        "ALA": [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        "ARG": [3.0, 3.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
        "ASN": [2.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "ASP": [2.0, 0.0, 2.0, -1.0, 0.0, 0.0, 0.0, 1.0],
        "CYS": [1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        "GLN": [2.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "GLU": [2.0, 0.0, 2.0, -1.0, 0.0, 0.0, 0.0, 1.0],
        "GLY": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "HIS": [2.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        "ILE": [2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        "LEU": [2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        "LYS": [2.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
        "MET": [2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        "PHE": [3.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        "PRO": [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        "SER": [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "THR": [1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        "TRP": [3.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        "TYR": [3.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        "VAL": [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        "CAF": [1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        "CME": [1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        "CSS": [1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        "OCY": [1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        "KCX": [2.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
        "MSE": [2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        "PHD": [2.0, 0.0, 2.0, -1.0, 0.0, 0.0, 0.0, 1.0],
        "PTR": [3.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    },
    columns=[
        "size",
        "hbd",
        "hba",
        "charge",
        "aromatic",
        "aliphatic",
        "charge_pos",
        "charge_neg",
    ],
    orient="index",
)

amino_acid_to_int = {aa: idx for idx, aa in enumerate(sitealign_feature_lookup.index)}


# we precomputed kissim fps
def load_kissim(structure_id: int) -> pd.DataFrame:
    path = _DATA / "processed" / "kissim" / f"{structure_id}.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


PHYSICOCHEMICAL = ["size", "hbd", "hba", "charge", "aromatic", "aliphatic"]
STRUCTURAL = ["sco", "exposure", "hinge_region", "dfg_region", "front_pocket", "center"]


def add_kissim_fp(
    data: Data,
    fp: pd.DataFrame,
    subset: Optional[List[str]] = None,
) -> Data:
    if subset is not None:
        fp = fp[list(set(subset))]
    fp = torch.from_numpy(fp.values)
    data.kissim_fp = fp.unsqueeze(0)
    return data


known_residues = [
    "A",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "V",
    "W",
    "X",
    "Y",
    "_",
]
mapping = -torch.ones(max(ord(c) for c in known_residues) + 1, dtype=torch.long)
mapping[[ord(c) for c in known_residues]] = torch.arange(len(known_residues))


def add_onehot_residues(data: Data, sequence: str):
    index = mapping[[ord(c) for c in sequence]]
    x = torch.zeros((len(sequence), len(known_residues) + 1))
    x[torch.arange(len(sequence)), index] = 1.0
    data[NodeType.PocketResidue].x = x
    return data
