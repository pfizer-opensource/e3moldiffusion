from collections import defaultdict
import logging
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

import torch
import numpy as np
from torch_scatter import scatter_add
from torch_geometric.data import HeteroData
from torch_geometric.utils import to_undirected
import pandas as pd

from kinodata.data.io import read_klifs_mol2
from kinodata.types import NodeType, RelationType
from .residue import sitealign_feature_lookup, amino_acid_to_int
from rdkit.Chem import GetPeriodicTable

pt = GetPeriodicTable()


def prepare_pocket_information(
    pocket_mol2_file: Union[str, Path],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # prepare pocket data frame
    # each row corresponds to one atom
    df_atom, df_bond = read_klifs_mol2(pocket_mol2_file)
    # TODO: are all residue names exactly 3 characters long??
    df_atom["residue.type"] = df_atom["residue.subst_name"].apply(lambda s: s[:3])

    # zero all indices
    df_atom["atom.id"] -= 1
    df_bond["source_atom_id"] -= 1
    df_bond["target_atom_id"] -= 1

    # select unique residues and featurize them
    # by looking up sitealign features based on residue type
    # lookup table taken from KiSSim
    df_residue = (
        df_atom[["residue.subst_id", "residue.type"]]
        .drop_duplicates()
        .set_index("residue.subst_id")
    )
    # add categorical integer feature for residue type
    df_residue["residue.type_idx"] = df_residue["residue.type"].apply(
        amino_acid_to_int.get
    )

    # set categorical integer feature for atom type / element
    # based on atomic numbers
    df_atom["atom.atomic_number"] = df_atom["atom.type"].apply(
        lambda t: pt.GetAtomicNumber(t.split(".")[0])
    )

    return df_atom, df_bond, df_residue


def remove_hydrogens(
    df_atom: pd.DataFrame, df_bond: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    is_h_mask = df_atom["atom.type"].apply(lambda t: "H" in t)
    is_h = torch.tensor(is_h_mask.values).float()
    row = torch.tensor(df_bond["source_atom_id"].values)
    col = torch.tensor(df_bond["target_atom_id"].values)

    num_hydrogens = torch.zeros(df_atom.index.max() + 1).float()
    scatter_add(is_h[row], col, 0, num_hydrogens)
    scatter_add(is_h[col], row, 0, num_hydrogens)
    df_atom["atom.num_hs"] = num_hydrogens.numpy()

    is_h_bond = is_h_mask[row.numpy()].values | is_h_mask[col.numpy()].values

    num_heavy = int((~is_h_mask).sum())
    relabeling = np.empty(df_atom.shape[0], dtype=int)
    relabeling[~is_h_mask] = np.arange(num_heavy)

    df_atom = df_atom[~is_h_mask]
    df_bond = df_bond[~is_h_bond]
    df_atom.loc[:, "atom.id"] = relabeling[df_atom["atom.id"].values]
    df_bond.loc[:, "source_atom_id"] = relabeling[df_bond["source_atom_id"].values]
    df_bond.loc[:, "target_atom_id"] = relabeling[df_bond["target_atom_id"].values]

    return df_atom, df_bond


def longrange_to_onehot(
    data: pd.DataFrame, column_key: str, min: int, max: int
) -> torch.Tensor:
    col = data[column_key]
    index = torch.tensor(col.values, dtype=torch.long) - min
    other = max + 1
    index[index < min] = other
    index[index > max] = other
    encoding = torch.zeros(index.shape[0], max - min + 2)
    encoding[torch.arange(index.size(0)), index - min] = 1.0
    return encoding


def any_to_onehot(
    data: pd.DataFrame,
    column_key: str,
    known_values: Sequence[Any],
) -> torch.Tensor:
    mapping = defaultdict(lambda: len(known_values))
    for index, item in enumerate(known_values):
        mapping[item] = index
    index = data[column_key].apply(mapping.get).values
    encoding = torch.zeros(index.shape[0], len(known_values) + 1)
    encoding[torch.arange(index.shape[0]), index] = 1.0
    return encoding


def add_pocket_information(
    data: HeteroData,
    pocket_mol2_file: Union[str, Path],
    residue_feature_subset: Optional[List[str]] = [
        "hbd",
        "hba",
        "aromatic",
        "aliphatic",
        "charge_pos",
        "charge_neg",
    ],
    residue_only: bool = True,
) -> Optional[HeteroData]:
    """TODO
    This is kind of a mess and needs to be refactored.

    Parameters
    ----------
    data : HeteroData
        heterogeneous pyg data object
    pocket_mol2_file : Union[str, Path]
        path to KLIFS pocket mol2 file
    residue_feature_subset : Optional[List[str]], optional
        which kissim features to use for featurizing residues,
        by default [ "hbd", "hba", "aromatic", "aliphatic", "charge_pos", "charge_neg", ]

    Returns
    -------
    Optional[HeteroData]
        The modified data object with pocket information
        or None if a known exception is caught.
    """
    df_atom, df_bond, df_residue = prepare_pocket_information(pocket_mol2_file)
    df_atom, df_bond = remove_hydrogens(df_atom, df_bond)

    # compute 3D positions of residues as
    # geometric center of member atoms.
    residue_centers = df_atom.groupby("residue.subst_id")[
        ["atom.x", "atom.y", "atom.z"]
    ].mean()

    try:
        residues_features = (
            sitealign_feature_lookup[residue_feature_subset]
            .loc[df_residue["residue.type"]]
            .set_index(df_residue.index)
        )
    except KeyError as e:
        logging.warning(f"Unable to map residue to sitealign features: {e}")
        return None

    # convert all features & positions to tensors and store them in the provided
    # pyg data object
    data[NodeType.PocketResidue].z = torch.tensor(
        df_residue["residue.type_idx"].values, dtype=torch.long
    )
    data[NodeType.PocketResidue].x = torch.tensor(
        residues_features.values, dtype=torch.float
    )
    data[NodeType.PocketResidue].pos = torch.tensor(
        residue_centers[["atom.x", "atom.y", "atom.z"]].values,
        dtype=torch.float,
    )

    if not residue_only:
        raise NotImplementedError
        # set element/atomic number feature
        data["pocket"].z = torch.tensor(
            df_atom["atom.atomic_number"].values, dtype=torch.long
        )

        # other features
        # formal charge
        charge_features = longrange_to_onehot(df_atom, "atom.charge", -2, 2)
        # num hydrogens
        h_features = longrange_to_onehot(df_atom, "atom.num_hs", 0, 4)
        data["pocket"].x = torch.cat((charge_features, h_features), dim=1)

        # atomic positions
        data["pocket"].pos = torch.tensor(
            df_atom[["atom.x", "atom.y", "atom.z"]].values,
            dtype=torch.float,
        )

        # covalent bond edge index
        row = torch.tensor(df_bond["source_atom_id"].values, dtype=torch.long)
        col = torch.tensor(df_bond["target_atom_id"].values, dtype=torch.long)
        edge_index = torch.stack((row, col))
        edge_attr = any_to_onehot(df_bond, "bond_type", ["1", "2", "3"])
        edge_index, edge_attr = to_undirected(edge_index, edge_attr)
        data["pocket", "bond", "pocket"].edge_index = edge_index
        data["pocket", "bond", "pocket"].edge_attr = edge_attr

        # store atom/residue member relation as edges
        # (a, r) if atom a belongs to residue r
        data["pocket_residue", "contains", "pocket"].edge_index = torch.tensor(
            df_atom[["residue.subst_id", "atom.id"]].values.T, dtype=torch.long
        )

    return data
