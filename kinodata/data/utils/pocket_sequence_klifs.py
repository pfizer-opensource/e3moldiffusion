from typing import Iterable, List
import requests
import json
import pandas as pd
from tqdm import tqdm


def get_pocket_sequence(structure_ids: Iterable[int]) -> List[str]:
    """Retrieve pocket sequences from KLIFS based on structure ID.

    Parameters
    ----------
    structure_ids : Iterable[int]
        Iterable over structure IDs.

    Returns
    -------
    List[str]
        List of corresponding pocket sequences.
    """
    sequences = []
    for structure_id in structure_ids:
        resp = requests.get(
            "https://klifs.net/api_v2/structure_list",
            params={"structure_ID": structure_id},
        )
        resp.raise_for_status()
        pocket_seq = json.loads(resp.content)[0]["pocket"]
        sequences.append(pocket_seq)
    return sequences


def add_pocket_sequence(
    target: pd.DataFrame,
    structure_id_key: str = "similar.klifs_structure_id",
    pocket_sequence_key: str = "structure.pocket_sequence",
) -> pd.DataFrame:
    """Given a dataframe with KLIFS structures as entries,
    add/extend it with an additional column that stores corresponding
    pocket sequences.

    Will query KLIFS based on structure ID.

    Parameters
    ----------
    target : pd.DataFrame
        The target data frame that should be modified.
    structure_id_key : str, optional
        The name of the column that stores structure IDs.
    pocket_sequence_key : str, optional
        The name of the newly created column that will store pocket sequences.

    Returns
    -------
    pd.DataFrame
        New data frame with modifications as descrived above.
    """
    klifs_structure_id = target[structure_id_key].unique()
    seq_raw = get_pocket_sequence(
        tqdm(klifs_structure_id, desc="Retrieving pocket sequences from KLIFS...")
    )
    df_seq = pd.DataFrame(
        {pocket_sequence_key: seq_raw, structure_id_key: klifs_structure_id}
    )
    extended = target.merge(df_seq, how="left", on=structure_id_key)
    return extended
