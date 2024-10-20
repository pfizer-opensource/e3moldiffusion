from collections import defaultdict
from typing import Dict, Optional, Tuple
import torch
from torch import Tensor

from torch_geometric.transforms import BaseTransform
from torch_geometric.data import HeteroData
from torch_geometric.utils import (
    remove_self_loops,
    to_dense_adj,
    to_undirected,
    coalesce,
)
from torch_cluster import radius
from itertools import product

from kinodata.types import NodeType, EdgeType, RelationType


def interactions_and_distances(
    pos_x: Tensor,
    pos_y: Optional[Tensor] = None,
    batch_x: Optional[Tensor] = None,
    batch_y: Optional[Tensor] = None,
    r: float = 1.0,
    max_num_neighbors: int = 32,
) -> Tuple[Tensor, Tensor]:
    if pos_y is None:
        pos_y = pos_x
        batch_y = batch_x
    y_ind, x_ind = radius(
        pos_x,
        pos_y,
        r,
        batch_x=batch_x,
        batch_y=batch_y,
        max_num_neighbors=max_num_neighbors,
    )
    dist = (pos_x[x_ind] - pos_y[y_ind]).pow(2).sum(dim=1).sqrt()
    edge_index = torch.stack((x_ind, y_ind))
    return edge_index, dist


class AddDistancesAndInteractions(BaseTransform):
    def __init__(
        self,
        default_radius: float = 5.0,
        edge_types: Optional[Dict[Tuple[NodeType, NodeType], float]] = None,
        distance_key: str = "edge_weight",
        max_num_neighbors: int = 85,
    ) -> None:
        super().__init__()
        self.distance_key = distance_key
        self.subset = None
        self.max_num_neighbors = max_num_neighbors
        self.default_radius = default_radius
        self.radii = defaultdict(lambda: default_radius)
        if edge_types:
            self.subset = set()
            for (u, v), r in edge_types.items():
                self.subset.add((u, v))
                self.subset.add((v, u))
                self.radii[(u, v)] = r
                self.radii[(v, u)] = r

    def __call__(self, data: HeteroData) -> HeteroData:
        if isinstance(data, HeteroData):
            node_types, edge_types = data.metadata()
            itr = product(node_types, node_types)
            if self.subset:
                itr = filter(lambda nt_pair: nt_pair in self.subset, itr)
            for nt_a, nt_b in itr:
                relation_key = (
                    RelationType.Intraacts if nt_a == nt_b else RelationType.Interacts
                )
                radius = self.radii[(nt_a, nt_b)]
                edge_index, dist = interactions_and_distances(
                    data[nt_a].pos,
                    data[nt_b].pos,
                    r=radius,
                    max_num_neighbors=self.max_num_neighbors,
                )
                if nt_a == nt_b:
                    num_nodes = data[nt_a].num_nodes
                    edge_index, dist = to_undirected(
                        edge_index, dist, num_nodes, reduce="min"
                    )
                    edge_index, dist = remove_self_loops(edge_index, dist)

                data[nt_a, relation_key, nt_b].edge_index = edge_index
                setattr(
                    data[nt_a, relation_key, nt_b],
                    self.distance_key,
                    dist.view(-1, 1).to(torch.float32),
                )

            return data
        else:
            raise NotImplementedError
            ...

    def __repr__(self) -> str:
        subset_str = "" if self.subset is None else f", subset={self.subset}"
        return f"{self.__class__.__name__}({self.radii})"


class ForceSymmetricInteraction(BaseTransform):
    def __init__(self, edge_type: Tuple[str, str, str]) -> None:
        super().__init__()
        self.edge_type = edge_type
        nt_a, relation, nt_b = self.edge_type
        assert nt_a != nt_b
        assert relation == "interacts"

    def _reverse_direction(self, edge_index):
        row, col = edge_index
        return torch.stack((col, row))

    def __call__(self, data: HeteroData) -> HeteroData:

        nt_a, relation, nt_b = self.edge_type

        for source, target in ([nt_a, nt_b], [nt_b, nt_a]):
            edge_index_1 = data[source, relation, target].edge_index
            dist_1 = data[source, relation, target].dist
            edge_index_2 = data[target, relation, source].edge_index
            dist_2 = data[target, relation, source].dist

            merged_edge_index = torch.cat(
                (edge_index_1, self._reverse_direction(edge_index_2)), dim=1
            )
            merged_dist = torch.cat((dist_1, dist_2))
            merged_edge_index, merged_dist = coalesce(
                merged_edge_index, merged_dist, reduce="mean"
            )

            data[source, relation, target].edge_index = merged_edge_index
            data[source, relation, target].dist = merged_dist

        return data


class AddDistances(BaseTransform):
    def __init__(self, edge_type: EdgeType, distance_key: str = "edge_weight") -> None:
        super().__init__()
        self.edge_type = edge_type
        self.distance_key = distance_key

    def __call__(self, data: HeteroData) -> HeteroData:
        row, col = data[
            self.edge_type[0], self.edge_type[1], self.edge_type[2]
        ].edge_index
        distance = (
            (
                (data[self.edge_type[0]].pos[row] - data[self.edge_type[2]].pos[col])
                .pow(2)
                .sum(dim=1)
                .sqrt()
            )
            .view(-1, 1)
            .to(torch.float32)
        )
        setattr(
            data[self.edge_type[0], self.edge_type[1], self.edge_type[2]],
            self.distance_key,
            distance,
        )
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.edge_type})"
