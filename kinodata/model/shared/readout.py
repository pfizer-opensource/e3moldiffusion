from typing import Iterable, Dict

import torch.nn as nn
from torch import Tensor
from torch_geometric.nn.aggr import Aggregation
from torch_geometric.data import HeteroData

from kinodata.model.egnn import resolve_act

NodeType = str


class HeteroReadout(nn.Module):
    def __init__(
        self,
        node_types: Iterable[NodeType],
        node_aggregation: Aggregation,
        aggregation_out_channels: int,
        hidden_channels: int,
        out_channels: int,
        act: str = "elu",
        final_act: str = "none",
    ) -> None:
        super().__init__()
        self.node_types = list(node_types)
        act_fn = resolve_act(act)
        final_act_fn = resolve_act(final_act)
        self.node_aggregation = node_aggregation

        self.lins = nn.ModuleDict()
        for node_type in self.node_types:
            self.lins[node_type] = nn.Sequential(
                nn.Linear(aggregation_out_channels, hidden_channels, bias=False), act_fn
            )

        self.f_predict = nn.Sequential(
            nn.LayerNorm(hidden_channels),
            nn.Linear(hidden_channels, hidden_channels),
            act_fn,
            nn.Linear(hidden_channels, out_channels),
            final_act_fn,
        )

    def forward(
        self, node_embeddings: Dict[NodeType, Tensor], data: HeteroData
    ) -> Tensor:
        aggr = dict()
        for node_type, embeddings in node_embeddings.items():
            aggr[node_type] = self.lins[node_type](
                self.node_aggregation(embeddings, data[node_type].batch)
            )
        aggr = sum(aggr.values())
        return self.f_predict(aggr)
