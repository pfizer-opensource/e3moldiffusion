from typing import Optional
import torch
from torch import Tensor
from torch.nn.functional import silu
from torch.nn import LayerNorm, Linear, MultiheadAttention


class MultiheadAttentionBlock(torch.nn.Module):
    r"""The Multihead Attention Block (MAB) from the `"Set Transformer: A
    Framework for Attention-based Permutation-Invariant Neural Networks"
    <https://arxiv.org/abs/1810.00825>`_ paper
    Args:
        channels (int): Size of each input sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        norm (str, optional): If set to :obj:`False`, will not apply layer
            normalization. (default: :obj:`True`)
        dropout (float, optional): Dropout probability of attention weights.
            (default: :obj:`0`)
    """

    def __init__(
        self,
        channels: int,
        heads: int = 1,
        layer_norm: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.channels = channels
        self.heads = heads
        self.dropout = dropout

        self.attn = MultiheadAttention(
            channels,
            heads,
            batch_first=True,
            dropout=dropout,
        )
        self.lin = Linear(channels, channels)
        self.layer_norm1 = LayerNorm(channels) if layer_norm else None
        self.layer_norm2 = LayerNorm(channels) if layer_norm else None

    def reset_parameters(self):
        self.attn._reset_parameters()
        self.lin.reset_parameters()
        if self.layer_norm1 is not None:
            self.layer_norm1.reset_parameters()
        if self.layer_norm2 is not None:
            self.layer_norm2.reset_parameters()

    def forward(
        self,
        x: Tensor,
        y: Tensor,
        x_mask: Optional[Tensor] = None,
        y_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """"""
        if y_mask is not None:
            y_mask = ~y_mask

        out, _ = self.attn(x, y, y, y_mask, need_weights=False)

        if x_mask is not None:
            out[~x_mask] = 0.0

        out = out + x

        if self.layer_norm1 is not None:
            out = self.layer_norm1(out)

        out = out + silu(self.lin(out))

        if self.layer_norm2 is not None:
            out = self.layer_norm2(out)

        return out

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.channels}, "
            f"heads={self.heads}, "
            f"layer_norm={self.layer_norm1 is not None}, "
            f"dropout={self.dropout})"
        )


class SetAttentionBlock(torch.nn.Module):
    r"""The Set Attention Block (SAB) from the `"Set Transformer: A
    Framework for Attention-based Permutation-Invariant Neural Networks"
    <https://arxiv.org/abs/1810.00825>`_ paper
    .. math::
        \mathrm{SAB}(\mathbf{X}) = \mathrm{MAB}(\mathbf{x}, \mathbf{y})
    Args:
        channels (int): Size of each input sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        norm (str, optional): If set to :obj:`False`, will not apply layer
            normalization. (default: :obj:`True`)
        dropout (float, optional): Dropout probability of attention weights.
            (default: :obj:`0`)
    """

    def __init__(
        self,
        channels: int,
        heads: int = 1,
        layer_norm: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.mab = MultiheadAttentionBlock(channels, heads, layer_norm, dropout)

    def reset_parameters(self):
        self.mab.reset_parameters()

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        return self.mab(x, x, mask, mask)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.mab.channels}, "
            f"heads={self.mab.heads}, "
            f"layer_norm={self.mab.layer_norm1 is not None}, "
            f"dropout={self.mab.dropout})"
        )
