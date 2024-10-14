from typing import List, Optional, Union

import torch
from torch_geometric.data import HeteroData
from torch.nn import init
from torch.nn import (
    Parameter,
    Module,
    Linear,
    Dropout,
    LayerNorm,
    ModuleDict,
    ModuleList,
    init,
)
from torch import Tensor, cat

from kinodata.types import NodeEmbedding
from kinodata.model.resolve import resolve_act


class CategoricalEmbedding(Module):
    @property
    def out_channels(self) -> int:
        return self.weight.data.size(1)

    def __init__(
        self,
        node_type: str,
        hidden_chanels: int,
        category_key: str,
        max_num_categories: int = 100,
        # enable parameter sharing for different node types
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.node_type = node_type
        self.category_key = category_key
        self.weight = Parameter(
            torch.empty(max_num_categories, hidden_chanels), requires_grad=True
        )
        self.ln = LayerNorm(hidden_chanels)
        self.reset_parameters()

    def reset_parameters(self):
        self.ln.reset_parameters()
        init.normal_(self.weight)

    def forward(self, data: HeteroData) -> Tensor:
        category = getattr(data[self.node_type], self.category_key)
        return self.ln(self.weight[category])


class AtomTypeEmbedding(CategoricalEmbedding):
    def __init__(
        self,
        node_type: str,
        hidden_chanels: int,
        category_key: str = "z",
        max_num_categories: int = 100,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            node_type,
            hidden_chanels,
            category_key,
            max_num_categories,
            dropout=dropout,
        )


class FeatureEmbedding(Module):
    def __init__(
        self,
        node_type: str,
        in_channels: int,
        hidden_channels: int,
        act: str,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.node_type = node_type
        self.lin = Linear(in_channels, hidden_channels)
        self.act = resolve_act(act)
        self.dropout = Dropout(dropout)
        self.ln = LayerNorm(hidden_channels)
        self.out_channels = self.lin.out_features

    def forward(self, data: HeteroData) -> Tensor:
        return self.ln(self.dropout(self.act(self.lin(data[self.node_type].x))))


class CombineConcat(Module):
    def __init__(self, embeddings: List[Module]) -> None:
        super().__init__()
        self.embeddings = ModuleList(embeddings)
        combined_channels = sum(emb.out_channels for emb in embeddings)
        self.lin = Linear(combined_channels, embeddings[0].output_channels, bias=False)

    def forward(self, data: HeteroData) -> Tensor:
        return self.lin(cat([emb(data) for emb in self.embeddings]))


class CombineSum(Module):
    def __init__(self, embeddings: List[Module]) -> None:
        super().__init__()
        assert len(set([emb.out_channels for emb in embeddings])) == 1
        self.embeddings = ModuleList(embeddings)

    def forward(self, data: HeteroData) -> Tensor:
        return sum([emb(data) for emb in self.embeddings])


class HeteroEmbedding(Module):
    def __init__(
        self, combination_mode: str = "sum", **embeddings: Union[Module, List[Module]]
    ) -> None:
        super().__init__()
        if combination_mode == "cat":
            Combination = CombineConcat
        elif combination_mode == "sum":
            Combination = CombineSum
        else:
            raise ValueError(combination_mode)

        self.embeddings = ModuleDict()
        for node_type, embedding in embeddings.items():
            if isinstance(embedding, list):
                self.embeddings[node_type] = Combination(embedding)
            else:
                self.embeddings[node_type] = embedding

    def forward(self, data: HeteroData) -> NodeEmbedding:
        return {node_type: emb(data) for node_type, emb in self.embeddings.items()}
