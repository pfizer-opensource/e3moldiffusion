from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn
from torch_geometric.typing import OptTensor

from e3moldiffusion.convs import (
    EQGATGlobalEdgeConvFinal,
    EQGATLocalConvFinal,
    TopoEdgeConvLayer,
)
from e3moldiffusion.modules import LayerNorm, AdaptiveLayerNorm


class EQGATEdgeGNN(nn.Module):
    """_summary_
    EQGAT GNN Network updating node-level scalar, vectors and position features as well as edge-features.
    Args:
        nn (_type_): _description_
    """

    def __init__(
        self,
        hn_dim: Tuple[int, int] = (64, 16),
        edge_dim: Optional[int] = 16,
        cutoff_local: float = 5.0,
        num_layers: int = 5,
        latent_dim: Optional[int] = None,
        use_cross_product: bool = False,
        vector_aggr: str = "mean",
        fully_connected: bool = True,
        local_global_model: bool = False,
        recompute_radius_graph: bool = False,
        recompute_edge_attributes: bool = True,
        edge_mp: bool = False,
        p1: bool = True,
        use_pos_norm: bool = True,
    ):
        super(EQGATEdgeGNN, self).__init__()

        assert fully_connected
        assert not local_global_model

        self.num_layers = num_layers
        self.fully_connected = fully_connected
        self.local_global_model = local_global_model
        self.cutoff_local = cutoff_local
        self.recompute_radius_graph = recompute_radius_graph
        self.recompute_edge_attributes = recompute_edge_attributes
        self.p1 = p1

        self.sdim, self.vdim = hn_dim
        self.edge_dim = edge_dim

        convs = []

        for i in range(num_layers):
            ## second or second last layer
            # lb = (i == 1 or i == num_layers - 2)
            lb = (i % 2 == 0) and (i != 0)
            # new: every second layer
            edge_mp_select = lb & edge_mp
            convs.append(
                EQGATGlobalEdgeConvFinal(
                    in_dims=hn_dim,
                    out_dims=hn_dim,
                    edge_dim=edge_dim,
                    has_v_in=i > 0,
                    use_mlp_update=i < (num_layers - 1),
                    vector_aggr=vector_aggr,
                    use_cross_product=use_cross_product,
                    edge_mp=edge_mp_select,
                    use_pos_norm=use_pos_norm,
                )
            )

        self.convs = nn.ModuleList(convs)

        if latent_dim:
            norm_module = AdaptiveLayerNorm
        else:
            norm_module = LayerNorm

        self.norms = nn.ModuleList(
            [norm_module(dims=hn_dim, latent_dim=latent_dim) for _ in range(num_layers)]
        )

        self.reset_parameters()

    def reset_parameters(self):
        for conv, norm in zip(self.convs, self.norms):
            conv.reset_parameters()
            norm.reset_parameters()

    def calculate_edge_attrs(
        self, edge_index: Tensor, edge_attr: OptTensor, pos: Tensor, sqrt: bool = True
    ):
        source, target = edge_index
        r = pos[target] - pos[source]
        a = pos[target] * pos[source]
        a = a.sum(-1)
        d = torch.clamp(torch.pow(r, 2).sum(-1), min=1e-6)
        if sqrt:
            d = d.sqrt()
        r_norm = torch.div(r, (1.0 + d.unsqueeze(-1) if self.p1 else d.unsqueeze(-1)))
        edge_attr = (d, a, r_norm, edge_attr)
        return edge_attr

    def to_dense_edge_tensor(self, edge_index, edge_attr, num_nodes):
        E = torch.zeros(
            num_nodes,
            num_nodes,
            edge_attr.size(-1),
            device=edge_attr.device,
            dtype=edge_attr.dtype,
        )
        E[edge_index[0], edge_index[1], :] = edge_attr
        return E

    def from_dense_edge_tensor(self, edge_index, E):
        return E[edge_index[0], edge_index[1], :]

    def forward(
        self,
        s: Tensor,
        v: Tensor,
        p: Tensor,
        edge_index_local: Tensor,
        edge_attr_local: Tuple[Tensor, Tensor, Tensor, Tensor],
        edge_index_global: Tensor,
        edge_attr_global: Tuple[Tensor, Tensor, Tensor, Tensor],
        z: OptTensor = None,
        batch: Tensor = None,
        context: OptTensor = None,
    ) -> Dict:
        # edge_attr_xyz (distances, cosines, relative_positions, edge_features)
        # (E, E, E x 3, E x F)

        for i in range(len(self.convs)):
            edge_index_in = edge_index_global
            edge_attr_in = edge_attr_global

            if context is not None and i > 0:
                s = s + context
            s, v = self.norms[i](x={"s": s, "v": v, "z": z}, batch=batch)
            out = self.convs[i](
                x=(s, v, p),
                batch=batch,
                edge_index=edge_index_in,
                edge_attr=edge_attr_in,
            )

            s, v, p, e = out["s"], out["v"], out["p"], out["e"]
            # p = p - scatter_mean(p, batch, dim=0)[batch]
            if self.recompute_edge_attributes:
                edge_attr_global = self.calculate_edge_attrs(
                    edge_index=edge_index_global, pos=p, edge_attr=e, sqrt=True
                )

            e = edge_attr_global[-1]

        out = {"s": s, "v": v, "e": e, "p": p}

        return out


class EQGATLocalGNN(nn.Module):
    """_summary_
    EQGAT GNN Network updating node-level scalar, vectors.
    Args:
        nn (_type_): _description_
    """

    def __init__(
        self,
        hn_dim: Tuple[int, int] = (64, 16),
        edge_dim: Optional[int] = 16,
        cutoff_local: float = 5.0,
        num_layers: int = 5,
        use_cross_product: bool = False,
        vector_aggr: str = "mean",
    ):
        super(EQGATLocalGNN, self).__init__()

        self.num_layers = num_layers
        self.cutoff_local = cutoff_local

        self.sdim, self.vdim = hn_dim
        self.edge_dim = edge_dim

        convs = []

        for i in range(num_layers):
            convs.append(
                EQGATLocalConvFinal(
                    in_dims=hn_dim,
                    out_dims=hn_dim,
                    edge_dim=edge_dim,
                    has_v_in=i > 0,
                    use_mlp_update=i < (num_layers - 1),
                    vector_aggr=vector_aggr,
                    use_cross_product=use_cross_product,
                )
            )

        self.convs = nn.ModuleList(convs)
        self.norms = nn.ModuleList([LayerNorm(dims=hn_dim) for _ in range(num_layers)])

        self.reset_parameters()

    def reset_parameters(self):
        for conv, norm in zip(self.convs, self.norms):
            conv.reset_parameters()
            norm.reset_parameters()

    def forward(
        self,
        s: Tensor,
        v: Tensor,
        p: Tensor,
        edge_index_local: Tensor,
        edge_attr_local: Tuple[Tensor, Tensor, Tensor, Tensor],
        edge_index_global: Tensor,
        edge_attr_global: Tuple[Tensor, Tensor, Tensor, Tensor],
        batch: Tensor = None,
    ) -> Dict:
        # edge_attr_xyz (distances, cosines, relative_positions, edge_features)
        # (E, E, E x 3, E x F)

        for i in range(len(self.convs)):
            edge_index_in = edge_index_local
            edge_attr_in = edge_attr_local

            s, v = self.norms[i](x={"s": s, "v": v}, batch=batch)

            out = self.convs[i](
                x=(s, v, p),
                batch=batch,
                edge_index=edge_index_in,
                edge_attr=edge_attr_in,
            )
            s, v = out["s"], out["v"]

        out = {"s": s, "v": v}

        return out


class TopoEdgeGNN(nn.Module):
    def __init__(
        self,
        in_dim,
        edge_dim: Optional[int] = 16,
        num_layers: int = 5,
    ):
        super(TopoEdgeGNN, self).__init__()

        self.num_layers = num_layers
        self.in_dim = in_dim
        self.edge_dim = edge_dim

        convs = []

        for i in range(num_layers):
            convs.append(
                TopoEdgeConvLayer(
                    in_dim=in_dim, out_dim=in_dim, edge_dim=edge_dim, aggr="mean"
                )
            )

        self.convs = nn.ModuleList(convs)

        self.norms = nn.ModuleList(
            [LayerNorm(dims=(in_dim, None)) for _ in range(num_layers)]
        )

        self.reset_parameters()

    def reset_parameters(self):
        for conv, norm in zip(self.convs, self.norms):
            conv.reset_parameters()
            norm.reset_parameters()

    def forward(
        self, s: Tensor, edge_index: Tensor, edge_attr: Tensor, batch: Tensor
    ) -> Dict:
        for i in range(len(self.convs)):
            s, _ = self.norms[i](x={"s": s, "v": None}, batch=batch)
            out, edge_attr = self.convs[i](
                x=s, edge_index=edge_index, edge_attr=edge_attr
            )
        out = {"s": s, "e": edge_attr}

        return out
