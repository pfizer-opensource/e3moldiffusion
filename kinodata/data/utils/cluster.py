import pandas as pd
import numpy as np
from typing import Callable, Generic, Sequence, TypeVar

import sklearn.cluster as cluster

T = TypeVar("T")


class Clustering(Generic[T]):
    cluster_key: str = "cluster_index"
    unique_items: Sequence[T]

    def __call__(
        self,
        data: pd.DataFrame,
        column_key: str,
        fn_similarity: Callable[[Sequence[T]], np.ndarray],
    ) -> pd.DataFrame:
        """
        Wrapper around clustering.
        Used for clustering all unique values of a
        specific dataset column

        Parameters
        ----------
        data : pd.DataFrame
        column_key : str
        fn_similarity: function that computes pairwise similarities.

        Returns
        -------
        pd.DataFrame
            The clustered data, i.e., the input data frame
            with a new column that contains the cluster label
            for each data point.
        """
        # TODO change this such that unique data points are given as input
        # instead of entire data?

        # only cluster the unique data points
        self.unique_items: Sequence[T] = data[column_key].unique()
        self.similarities = fn_similarity(self.unique_items)
        cluster_labels = self.solve()
        df_cluster = pd.DataFrame(
            {
                column_key: self.unique_items,
                self.cluster_key: cluster_labels,
            }
        )
        return pd.merge(data, df_cluster, on=column_key)

    def solve(self) -> Sequence[int]:
        """
        Find a clustering based on the given similarities.

        Returns
        -------
        Sequence[int]
            The labels (cluster indices) for all data points.
        """
        raise NotImplementedError


class AffinityPropagation(Clustering):
    def __init__(self) -> None:
        super().__init__()
        self.model = cluster.AffinityPropagation(affinity="precomputed")

    def solve(self):
        self.model.fit(self.similarities)
        return self.model.labels_


# TODO implement hierarchical/agglomerative clustering for "easier" splits?
class HierarchicalClustering(Clustering):
    def __init__(self) -> None:
        super().__init__()
        raise NotImplementedError
