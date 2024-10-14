import pandas as pd
import numpy as np
from typing import Dict, Sequence, TypeVar, Collection, Protocol
from pathlib import Path


# NOTE DEPRECATED/UNUSED
def solve_random(
    dataset: pd.DataFrame,
    partition_key: str,
    target_split: Dict[str, float],
    seed: int = 0,
    balanced: bool = True,
    epsilon: float = 0.05,
) -> pd.DataFrame:
    """Roughly split a dataset into splits of given sizes via
    a random selection heuristic.

    The data is partioned with respect to some attribute (see parameter 'partition_key') and
    entire partition subsets are assigned to splits randomly
    while trying to maintain the given size constraints.

    More specifically, the partition subsets are processed in a fixed order
    and each processed subset is assigned to some split with probability
    `q` where `q` is a value between zero and one equal to the given target
    split size. If adding the partition subset to the chosen split would
    exceed the target size significantly (see parameter 'epsilon'), an alternative split
    is chosen.

    Parameters
    ----------
    dataset : pd.DataFrame
        The dataset, represented as a Pandas data frame. One row
        corresponds to one data point.
    partition_key : str
        The key/column name that indexes the partioning attribute,
        for example scaffold, or scaffold cluster indices.
    target_split : Dict[str, float]
        The desired splits and sizes. Example:
        ```
        target_split = {'train': 0.9, 'test': 0.1}
        ```
    seed : int, optional
        Random seed, by default 0.
    balanced : bool, optional
        Sort the partition by subset sizes and select randomly in descending order, by default True
        This makes it more likely for large partition subsets to be assigned to the largest split
        (typically the training data split).
    epslon : float, optional
        If a splits size would exceed its' target split size plus epsilon times the number of total
        data points after adding a chosen partition subset, add the partition subset to
        a randomly chosen alternative split, where this is not the case, instead.

        If no such alterantives exist, the procedure fails and a RuntimeError is raised.

    Returns
    -------
    pd.DataFrame
        The original data frame, but with an added column named 'split' that partitions the
        data into splits of roughly the given size.
    """
    # input preprocessing
    split_names, split_ratios = list(target_split.keys()), list(target_split.values())
    assert np.isclose(sum(split_ratios), 1.0), "Target split sizes must sum up to 1."

    # perform partitioning/grouping
    assert (
        dataset[partition_key].notna().all()
    ), "Cannot partition data points with nan values at partition key"
    groups = dataset.groupby(partition_key)

    # more input preprocessing / setup some helpful data structures
    group_sizes = groups.size()
    num_total = group_sizes.sum()
    epsilon = int(epsilon * num_total)
    target_split_sizes = {
        split_name: int(ratio * num_total) for split_name, ratio in target_split.items()
    }
    current_split_sizes = {split_name: 0 for split_name in split_names}

    # helper function for better readability
    def exceeds_split_capacity(added_size: int, split_name: str) -> bool:
        return (
            current_split_sizes[split_name] + added_size
            > target_split_sizes[split_name] + epsilon
        )

    # create the order in which the partition is processed
    group_sequence = group_sizes
    if balanced:
        group_sequence = group_sequence.sort_values(ascending=False)

    # seed a random number generator and
    # carry out the assignment procedure
    rng = np.random.default_rng(seed)
    split = {}
    for group, size in group_sequence.items():
        split_name = rng.choice(split_names, p=split_ratios)
        if exceeds_split_capacity(size, split_name):
            alternatives = [
                split_name
                for split_name in split_names
                if not exceeds_split_capacity(size, split_name)
            ]
            if len(alternatives) == 0:
                raise RuntimeError(
                    "Unable to find a split that satisfies the given constraints."
                    "Try again with a larger value of epsilon or a different random seed."
                )
            split_name = rng.choice(alternatives)

        # add partition subset (group) to chosen split
        # and update split size
        current_split_sizes[split_name] += size
        split[group] = split_name

    # modify and return the dataset
    # the split assignment is added as a new column
    split = pd.DataFrame(
        {partition_key: list(split.keys()), "split": list(split.values())}
    )
    split_dataset = pd.merge(dataset, split, on=partition_key)
    return split_dataset
