from pathlib import Path
from functools import singledispatchmethod
from typing import List, Optional, Protocol
import numpy as np
from numpy.random import default_rng
from sklearn.model_selection import GroupKFold, KFold
import pandas as pd

from .data_split import Split
from .utils.cluster import AffinityPropagation
from .utils.similarity import BLOSUMSubstitutionSimilarity
from .dataset import KinodataDocked


def _split_random(a: np.ndarray, percentile: float, seed: int = 0):
    rng = default_rng(seed)
    pivot_ = int(a.shape[0] * percentile)
    permuted = rng.permutation(a)
    return permuted[:pivot_], permuted[pivot_:]


def _generator_to_list(generator):
    splits = [
        Split(train_index, *_split_random(test_index, 0.5))  # type: ignore
        for train_index, test_index in generator
    ]
    return splits


def group_k_fold_split(
    group_index: np.ndarray,
    k: int,
) -> List[Split]:
    group_k_fold = GroupKFold(k)
    _X = np.zeros((group_index.shape[0], 1))
    generator = group_k_fold.split(_X, groups=group_index)
    return _generator_to_list(generator)


def random_k_fold_split(data_index: np.ndarray, k: int) -> List[Split]:
    k_fold = KFold(k, shuffle=True)
    generator = k_fold.split(data_index)
    return _generator_to_list(generator)


class KinodataKFoldSplit:
    """
    Wraps all supported splitting methods.
    and implements dataset-dependent caching.
    Splits are cached in a datasets processed dir.
    """

    pocket_clustering = AffinityPropagation()
    pocket_similarity_measure = BLOSUMSubstitutionSimilarity

    def __init__(self, split_type: str, k: int) -> None:
        assert split_type in (
            "scaffold-k-fold",
            "pocket-k-fold",
            "random-k-fold",
        ), f"Unknown split type {split_type}"
        self.k = k
        self.split_type = split_type

    @singledispatchmethod
    def cache_dir(self, dataset) -> Path:
        raise NotImplementedError(f"cahe_dir({type(dataset)})")

    @cache_dir.register
    def _(self, dataset: KinodataDocked) -> Path:
        return Path(dataset.processed_dir) / self.split_type

    @cache_dir.register
    def _(self, dataset_dir: Path) -> Path:
        return dataset_dir / self.split_type

    def split_files(
        self,
        dataset: Optional[KinodataDocked] = None,
        dataset_dir: Optional[Path] = None,
    ) -> List[Path]:
        if dataset_dir is not None:
            cache_dir = self.cache_dir(dataset_dir)
        if dataset is not None:
            cache_dir = self.cache_dir(dataset)
        assert cache_dir is not None
        return [cache_dir / f"{i}:{self.k}.csv" for i in range(1, self.k + 1)]

    def _split(self, dataset: KinodataDocked):
        if self.split_type == "scaffold-k-fold":
            scaffolds, idents = zip(*[(data.scaffold, data.ident) for data in dataset])
            scaffolds = np.array(scaffolds)
            splits = group_k_fold_split(group_index=scaffolds, k=self.k)
            return splits
        if self.split_type == "pocket-k-fold":
            pocket_data = pd.DataFrame(
                {
                    "index": np.arange(len(dataset.data.pocket_sequence)),
                    "pocket_sequence": dataset.data.pocket_sequence,
                }
            )
            df_cluster_labels = self.pocket_clustering(
                pocket_data,
                "pocket_sequence",
                fn_similarity=self.pocket_similarity_measure(),
            ).sort_values(by="index", ascending=True)
            assert df_cluster_labels.shape[0] == pocket_data.shape[0]
            return group_k_fold_split(
                np.array(df_cluster_labels[self.pocket_clustering.cluster_key].values),
                k=self.k,
            )
        if self.split_type == "random-k-fold":
            idents = [data.ident for data in dataset]
            return random_k_fold_split(idents, self.k)

    def split(self, dataset: KinodataDocked) -> List[Split]:
        split_files = self.split_files(dataset)
        if all(f.exists() for f in split_files):
            return [Split.from_csv(f) for f in split_files]

        splits: List[Split] = self._split(dataset)

        for split, f in zip(splits, split_files):
            if not f.parents[0].exists():
                f.parents[0].mkdir()
            split.to_data_frame().to_csv(f, index=False)
            split.source_file = str(f)

        return splits
