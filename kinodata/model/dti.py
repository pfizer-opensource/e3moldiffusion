from typing import Any, Callable, Optional, Protocol, Tuple

import torch
from kinodata.configuration import Config
from kinodata.data.featurization.bonds import NUM_BOND_TYPES
from kinodata.data.featurization.residue import known_residues
from kinodata.model.regression import RegressionModel
from kinodata.model.resolve import resolve_act, resolve_loss
from kinodata.model.shared import SetAttentionBlock
from kinodata.model.shared.gine import LigandGINE
from kinodata.types import NodeType
from torch import Tensor
from torch.nn import Embedding, LayerNorm, Linear, Module, ModuleList, Sequential, SiLU
from torch_geometric.nn.pool import global_add_pool
from torch_geometric.utils import to_dense_batch


class Encoder(Protocol):
    def __call__(self, batch: Any) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Parameters
        ----------
        batch : Any
            pyg data object

        Returns
        -------
        Tuple[Tensor, Optional[Tensor]]
            Embeddings of size (N, d) (float)
            Index tensor of size (N,) (long), assigns embeddings to their batch.
        """
        ...


class Decoder(Protocol):
    def __call__(
        self,
        ligand_embeddings: Tensor,
        pocket_embeddings: Tensor,
        ligand_batch: Optional[Tensor] = None,
        pocket_batch: Optional[Tensor] = None,
    ) -> Tensor:
        ...


class DTIModel(RegressionModel):
    def __init__(
        self,
        config: Config,
        ligand_encoder_cls: Callable[..., Encoder],
        pocket_encoder_cls: Callable[..., Encoder],
        decoder_cls: Callable[..., Decoder],
    ) -> None:
        super().__init__(config)
        self.criterion = resolve_loss(config.loss_type)
        self.ligand_encoder = config.init(ligand_encoder_cls)
        self.pocket_encoder = config.init(pocket_encoder_cls)
        self.decoder = config.init(decoder_cls)

    def forward(self, batch) -> Tensor:
        x_ligand, batch_ligand = self.ligand_encoder(batch)
        x_pocket, batch_pocket = self.pocket_encoder(batch)
        prediction = self.decoder(x_ligand, x_pocket, batch_ligand, batch_pocket)
        return prediction


class ResidueTransformer(Module):
    def __init__(
        self,
        residue_size: int,
        hidden_channels: int,
        num_attention_blocks: int,
        num_heads: int = 1,
    ) -> None:
        super().__init__()
        self.lin = Sequential(
            Linear(residue_size, hidden_channels), SiLU(), LayerNorm(hidden_channels)
        )
        self.positional_encoding = Embedding(85, hidden_channels)
        self.attention_blocks = ModuleList(
            [
                SetAttentionBlock(hidden_channels, num_heads)
                for _ in range(num_attention_blocks)
            ]
        )

    def get_residue_representation(self, batch):
        x, _ = to_dense_batch(
            batch[NodeType.PocketResidue].x, batch[NodeType.PocketResidue].batch
        )
        return x

    def forward(self, batch) -> Tuple[Tensor, Optional[Tensor]]:
        x = (
            self.lin(self.get_residue_representation(batch))
            + self.positional_encoding.weight
        )
        for attn in self.attention_blocks:
            x = attn(x)
        return x, None


class KissimTransformer(ResidueTransformer):
    def __init__(
        self,
        residue_size: int,
        hidden_channels: int,
        num_attention_blocks: int,
        num_heads: int = 1,
    ) -> None:
        super().__init__(residue_size, hidden_channels, num_attention_blocks, num_heads)

    def get_residue_representation(self, batch):
        return batch.kissim_fp.float()


def _sum_aggr(
    x: Tensor, index: Optional[Tensor] = None, feature_dim: int = 1
) -> Tensor:
    if index is not None:
        return global_add_pool(x, index)
    else:
        return x.sum(dim=feature_dim)


class GlobalSumDecoder(Module):
    def __init__(
        self,
        hidden_channels: int,
        out_channels: int = 1,
        act: str = "silu",
        feature_dim: int = -1,
    ) -> None:
        super().__init__()
        self.act = resolve_act(act)
        self.f_ligand = Sequential(
            Linear(hidden_channels, hidden_channels),
            self.act,
            LayerNorm(hidden_channels),
        )
        self.f_pocket = Sequential(
            Linear(hidden_channels, hidden_channels),
            self.act,
            LayerNorm(hidden_channels),
        )
        self.f_combined = Sequential(
            Linear(hidden_channels * 2, hidden_channels),
            self.act,
            Linear(hidden_channels, out_channels),
        )
        self.feature_dim = feature_dim

    def forward(
        self,
        ligand_embeddings: Tensor,
        pocket_embeddings: Tensor,
        ligand_batch: Optional[Tensor] = None,
        pocket_batch: Optional[Tensor] = None,
    ):
        ligand_repr = self.f_ligand(_sum_aggr(ligand_embeddings, ligand_batch))
        pocket_repr = self.f_pocket(_sum_aggr(pocket_embeddings, pocket_batch))
        combined_repr = torch.cat((ligand_repr, pocket_repr), dim=self.feature_dim)
        return self.f_combined(combined_repr)


class ResidueFeaturization:
    Kissim = "kissim"
    Onehot = "onehot"


def make_model(config):
    if config.residue_featurization == ResidueFeaturization.Onehot:
        ResidueModel = ResidueTransformer
        config["residue_size"] = len(known_residues) + 1
    elif config.residue_featurization == ResidueFeaturization.Kissim:
        ResidueModel = KissimTransformer
        config["residue_size"] = 6
    else:
        raise ValueError(config.residue_featurization)
    return DTIModel(config, LigandGINE, ResidueModel, GlobalSumDecoder)
