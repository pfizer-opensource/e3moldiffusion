from typing import Optional, Tuple
from torch import Tensor, tensor
from torch.nn import Module, Linear, LayerNorm, Sequential, Identity, Parameter
from torch_geometric.utils import softmax
from torch_scatter import scatter_add

OptTensor = Optional[Tensor]


class SparseAttention(Module):
    """
    Sparse Query/Key/Value attention.
    """

    hidden_channels: int = 8
    num_heads: int = 4

    def __init__(
        self, hidden_channels: int, num_heads: int, interaction_bias: bool = True
    ) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.projected_channels = hidden_channels // num_heads
        self.num_heads = num_heads
        self.interaction_bias = interaction_bias
        self.lin_query = Linear(
            self.hidden_channels, self.projected_channels * self.num_heads, bias=False
        )
        self.lin_key_value = Linear(
            self.hidden_channels,
            self.projected_channels * self.num_heads * 2,
            bias=False,
        )
        if self.interaction_bias:
            self.lin_bias = Linear(
                self.hidden_channels,
                self.projected_channels * self.num_heads * 2,
                bias=False,
            )
        self.lin_out = Linear(
            self.projected_channels * self.num_heads, self.hidden_channels, bias=False
        )
        self.normalizer = Parameter(
            tensor([self.projected_channels]).float().sqrt(),
            requires_grad=True,
        )

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        query_index: Tensor,
        key_index: Tensor,
        interaction_repr: OptTensor = None,
    ) -> Tuple[Tensor, OptTensor]:
        """
        Perform sparse attention based on `M` Query/Key interactions.

        Interactions 0...M-1 are encoded by the aligned index tensors
        `query_index` and `key_index`.

        Interactions may have their own representation that biases attention.


        Parameters
        ----------
        query : Tensor
            (N, d)
        key : Tensor
            (N, d)
        query_index : Tensor
            (M,)
        key_index : Tensor
            (M,)
        interaction_repr : Tensor | None
            (M, d)
        """
        N = query.size(0)
        M = query_index.size(0)
        d = self.projected_channels
        H = self.num_heads

        # (N, d) -> (N, d, H)
        query = self.lin_query(query).view(N, d, H)
        key, value = self.lin_key_value(key).chunk(2, dim=1)
        key = key.view(N, d, H)
        value = value.view(N, d, H)

        # (M, d, H)
        query = query[query_index]
        key = key[key_index]
        value = value[key_index]

        if self.interaction_bias:
            # (M, d) -> ((M, d * H)
            bias_mul, bias_add = self.lin_bias(interaction_repr).chunk(2, dim=1)

            # (M, d, H)
            bias_mul = bias_mul.view(M, d, H)
            bias_add = bias_add.view(M, d, H)

            # (M, d, H) -> (M, H)
            attention_logit = (query * (key * (1 + bias_mul) + bias_add)).sum(
                dim=1
            ) / self.normalizer.clamp(min=1.0, max=self.projected_channels)
            pair_logits = None
        else:
            attention_logit = (query * key).sum(dim=1) / self.normalizer.clamp(
                min=1.0, max=self.projected_channels
            )

        attention = softmax(attention_logit, query_index, dim=0)
        result = scatter_add(attention.unsqueeze(1) * value, query_index, 0)
        result = result.view(N, -1)
        result = self.lin_out(result)
        pair_logits = attention_logit.view(M, -1)
        return result, pair_logits


class SPAB(Module):
    """
    Graph self-attention with sparse topology and explicit interaction representations.
    """

    def __init__(
        self,
        hidden_channels: int,
        num_heads: int,
        act: Module,
        ln1: bool = True,
        ln2: bool = False,
        ln3: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        hidden_channels : int
            Hidden dimensionality
        num_heads : int
            Number of head in multi-headed attention
        act : Module
            Activation function
        ln1 : bool, optional
            Apply layer normalization after first residual connection, by default True
        ln2 : bool, optional
            Apply layer normalization after node-wise FFN, by default False
        ln3 : bool, optional
            Apply layer normalization to updated edge representation (after edge-wise FFN), by default True
        """
        super().__init__()
        self.attention = SparseAttention(hidden_channels, num_heads)
        self.ln1 = LayerNorm(hidden_channels) if ln1 else Identity()
        self.ff = Sequential(
            Linear(hidden_channels, hidden_channels),
            act,
            Linear(hidden_channels, hidden_channels),
        )
        self.ln2 = LayerNorm(hidden_channels) if ln2 else Identity()

        self.ff_edge = Sequential(
            Linear(num_heads, hidden_channels),
            act,
            Linear(hidden_channels, hidden_channels),
        )
        self.ln3 = LayerNorm(hidden_channels) if ln3 else Identity()

    def forward(
        self, x: Tensor, edge_repr: Tensor, edge_index: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Apply sparse self-attention to input graph node embeddings.
        Attention is biased based on the representation of edges.

        Parameters
        ----------
        x : Tensor
            Node embeddings. Shape (N, d)
        edge_repr : Tensor
            Edge/Interaction embedding. Shape (M, d)
        edge_index : Tensor
            Interactions between nodes in COO forma. Shape (2, M).
            Long tensor with entries in (0,1,...,N-1).

        Returns
        -------
        Tuple[Tensor, Tensor]
            Updated node and edge embedding/representation.
        """
        x_, pair_logits = self.attention(x, x, edge_index[0], edge_index[1], edge_repr)
        x = self.ln1(x_ + x)
        x = self.ln2(self.ff(x) + x)
        edge_repr = self.ln3(self.ff_edge(pair_logits) + edge_repr)
        return x, edge_repr
