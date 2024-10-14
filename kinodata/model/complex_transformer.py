from functools import partial
from typing import List, Optional, Tuple
import torch
from torch import Tensor
from torch.nn import (
    ModuleList,
    Embedding,
    Sequential,
    Linear,
    Module,
    Parameter,
    Dropout,
    BatchNorm1d,
)
from torch_geometric.data import HeteroData
from torch_geometric.nn.norm import GraphNorm
from torch_geometric.nn.aggr import SoftmaxAggregation
from torch_geometric.utils import coalesce
from torch_cluster import knn_graph

from kinodata.configuration import Config

from ..types import NodeEmbedding, NodeType, RelationType
from .shared.dist_embedding import GaussianDistEmbedding
from .sparse_transformer import SPAB
from .regression import RegressionModel
from .resolve import resolve_act
from ..data.featurization.atoms import AtomFeatures
from ..data.featurization.bonds import NUM_BOND_TYPES

OptTensor = Optional[Tensor]


def FF(din, dout, act):
    return Sequential(Linear(din, dout), act)


class InteractionModule(Module):
    hidden_channels: int

    def __init__(
        self, hidden_channels: int, act: str = "none", bias: bool = False
    ) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.act = resolve_act(act)
        self.bias = Parameter(torch.zeros(self.hidden_channels), requires_grad=bias)

    def out(self, edge_repr: Tensor) -> Tensor:
        return self.act(edge_repr + self.bias)

    def interactions(self, data: HeteroData) -> Tuple[Tensor, OptTensor, OptTensor]:
        ...

    def process_attr(self, edge_attr: Tensor) -> Tensor:
        ...

    def process_weight(self, edge_weight: Tensor) -> Tensor:
        ...

    def forward(self, data: HeteroData) -> Tuple[Tensor, Tensor]:
        edge_index, edge_attr, edge_weight = self.interactions(data)
        device = [x.device for x in (edge_attr, edge_weight) if x is not None][0]
        edge_repr = torch.zeros(edge_index.size(1), self.hidden_channels, device=device)
        if edge_attr is not None:
            edge_repr = edge_repr + self.process_attr(edge_attr)
        if edge_weight is not None:
            edge_repr = edge_repr + self.process_weight(edge_weight)
        return edge_index, self.out(edge_repr)


class CovalentInteractions(InteractionModule):
    def __init__(
        self,
        hidden_channels: int,
        act: str = "none",
        bias: bool = False,
        node_type: str = None,
    ) -> None:
        super().__init__(hidden_channels, act, bias)
        assert node_type is not None
        self.node_type = node_type
        self.lin = Linear(NUM_BOND_TYPES, hidden_channels)

    def interactions(self, data: HeteroData) -> Tuple[Tensor, OptTensor, OptTensor]:
        edge_store = data[self.node_type, RelationType.Covalent, self.node_type]
        return edge_store.edge_index, edge_store.edge_attr, None

    def process_attr(self, edge_attr: Tensor) -> Tensor:
        return self.lin(edge_attr.float())


class StructuralInteractions(InteractionModule):
    def __init__(
        self,
        hidden_channels: int,
        act: str = "none",
        bias: bool = False,
        interaction_radius: float = 8.0,
        max_num_neighbors: int = 16,
        rbf_size: int = None,
    ) -> None:
        super().__init__(hidden_channels, act, bias)
        self.interaction_radius = interaction_radius
        self.max_num_neighbors = max_num_neighbors
        self.rbf_size = rbf_size if rbf_size else hidden_channels
        self.distance_embedding = GaussianDistEmbedding(rbf_size, interaction_radius)
        self.lin = Linear(rbf_size, hidden_channels, bias=False)

    def interactions(self, data: HeteroData) -> Tuple[Tensor, OptTensor, OptTensor]:
        pos = data[NodeType.Complex].pos
        batch = data[NodeType.Complex].batch
        edge_index = knn_graph(pos, self.max_num_neighbors + 1, batch, loop=True)
        distances = (pos[edge_index[0]] - pos[edge_index[1]]).pow(2).sum(dim=1).sqrt()
        mask = distances <= self.interaction_radius
        edge_index = edge_index[:, mask]
        distances = distances[mask]
        return edge_index, None, distances

    def process_weight(self, edge_weight: Tensor) -> Tensor:
        return self.distance_embedding(edge_weight)


class CombinedInteractions(Module):
    def __init__(self, interactions: List[InteractionModule], act: str) -> None:
        super().__init__()
        self.interactions = ModuleList(interactions)
        self.act = resolve_act(act)
        assert (
            len(set([intr.hidden_channels for intr in interactions])) == 1
        ), "Interactions should not map edge representation to different number of hidden channels."
        self.bias = Parameter(torch.zeros(interactions[0].hidden_channels))

    def forward(self, data: HeteroData) -> Tuple[Tensor, Tensor]:
        edge_indices, edge_reprs = zip(*[intr(data) for intr in self.interactions])
        edge_index, edge_repr = coalesce(
            torch.cat(edge_indices, 1), torch.cat(edge_reprs, 0), reduce="add"
        )
        return edge_index, self.act(edge_repr + self.bias)


class ComplexTransformer(RegressionModel):
    def __init__(
        self,
        config: Config,
        hidden_channels: int,
        num_heads: int,
        num_attention_blocks: int,
        interaction_radius: float,
        max_num_neighbors: int,
        act: str,
        max_atomic_number: int = 100,
        atom_attr_size: int = AtomFeatures.size,
        ln1: bool = True,
        ln2: bool = False,
        ln3: bool = True,
        graph_norm: bool = True,
        decoder_hidden_layers: int = 1,
        interaction_modes: List[str] = [],
        dropout: float = 0.1,
    ) -> None:
        super().__init__(config)
        assert len(config["node_types"]) == 1
        assert config["node_types"][0] == NodeType.Complex
        self.act = resolve_act(act)
        self.d_cut = interaction_radius
        self.max_num_neighbors = max_num_neighbors
        intr_bias = len(interaction_modes) == 1
        intr = []
        for mode in interaction_modes:
            if mode == "covalent":
                module = CovalentInteractions(
                    hidden_channels, act, intr_bias, config["node_types"][0]
                )
            elif mode == "structural":
                module = StructuralInteractions(
                    hidden_channels,
                    act,
                    intr_bias,
                    interaction_radius,
                    max_num_neighbors,
                    hidden_channels,
                )
            else:
                raise ValueError(mode)
            intr.append(module)
        if len(intr) > 1:
            self.interaction_module = CombinedInteractions(intr, act)
        else:
            self.interaction_module = intr[0]

        self.atomic_num_embedding = Embedding(max_atomic_number, hidden_channels)
        self.lin_atom_features = Linear(atom_attr_size, hidden_channels)
        self.attention_blocks = ModuleList(
            [
                SPAB(hidden_channels, num_heads, self.act, ln1, ln2, ln3)
                for _ in range(num_attention_blocks)
            ]
        )
        if graph_norm:
            self.norm_layers = ModuleList(
                [GraphNorm(hidden_channels) for _ in range(num_attention_blocks)]
            )
        else:
            self.norm_layers = [lambda x, b: x] * num_attention_blocks
        self.aggr = SoftmaxAggregation(learn=True, channels=hidden_channels)
        self.out = Sequential(
            *(
                [Dropout(dropout), BatchNorm1d(hidden_channels)]
                + [
                    FF(hidden_channels, hidden_channels, self.act)
                    for _ in range(decoder_hidden_layers)
                ]
                + [Linear(hidden_channels, 1)]
            )
        )

    def forward(self, data: HeteroData) -> NodeEmbedding:
        node_store = data[NodeType.Complex]
        node_repr = self.act(
            self.atomic_num_embedding(node_store.z)
            + self.lin_atom_features(node_store.x)
        )
        edge_index, edge_repr = self.interaction_module(data)
        for sparse_attention_block, norm in zip(
            self.attention_blocks, self.norm_layers
        ):
            node_repr, edge_repr = sparse_attention_block(
                node_repr, edge_repr, edge_index
            )
            node_repr = norm(node_repr, node_store.batch)

        graph_repr = self.aggr(node_repr, node_store.batch)
        return self.out(graph_repr)


def make_model(config: Config):
    cls = partial(ComplexTransformer, config)
    return config.init(cls)
