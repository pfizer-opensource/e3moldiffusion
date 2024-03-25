from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.nn import radius_graph
from torch_geometric.nn.inits import reset
from torch_geometric.typing import OptTensor

from e3moldiffusion.convs import (
    ConvLayer,
    EQGATConv,
    EQGATGlobalEdgeConvFinal,
    EQGATLocalConvFinal,
    TopoEdgeConvLayer,
)
from e3moldiffusion.modules import AdaptiveLayerNorm, DenseLayer, LayerNorm, SE3Norm

from torch_geometric.utils import remove_self_loops, sort_edge_index
from experiments.utils import get_edges

class EQGATEnergyGNN(nn.Module):
    def __init__(
        self,
        hn_dim: Tuple[int, int] = (256, 128),
        cutoff: float = 5.0,
        num_layers: int = 5,
        num_rbfs: int = 20,
        use_cross_product: bool = False,
        vector_aggr: str = "mean",
    ):
        super(EQGATEnergyGNN, self).__init__()
        self.sdim, self.vdim = hn_dim
        self.cutoff = cutoff
        self.num_layers = num_layers

        convs = []

        for i in range(num_layers):
            convs.append(
                EQGATConv(
                    in_dims=hn_dim,
                    out_dims=hn_dim,
                    num_rbfs=num_rbfs,
                    cutoff=cutoff,
                    has_v_in=i > 0,
                    use_mlp_update=i < (num_layers - 1),
                    vector_aggr=vector_aggr,
                    use_cross_product=use_cross_product,
                )
            )

        self.convs = nn.ModuleList(convs)

        self.norms = nn.ModuleList([LayerNorm(dims=hn_dim) for _ in range(num_layers)])

    def forward(
        self,
        s: Tensor,
        v: Tensor,
        edge_index: Tensor,
        edge_attr: Tuple[Tensor, Tensor],
        batch: Tensor = None,
    ) -> Dict:
        # edge_attr_xyz (distances, relative_positions)
        # (E, E x 3)
        for i in range(len(self.convs)):
            s, v = self.norms[i](x={"s": s, "v": v}, batch=batch)
            out = self.convs[i](
                x=(s, v),
                batch=batch,
                edge_index=edge_index,
                edge_attr=edge_attr,
            )
            s, v = out
        return s, v


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
        num_atom_features: int = 16,
        num_bond_types: int = 5,
        coords_param: str = "data",
        num_context_features: int = 0,
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
        use_out_norm: bool = True,
        property_prediction: bool = False,
        store_intermediate_coords: bool = False,
        ligand_pocket_interaction: bool = False,
        dynamic_graph: bool = False,
        kNN: Optional[int] = None,
        use_rbfs: bool = False,
        mask_pocket_edges: bool = False,
        model_edge_rbf_interaction: bool = False,
        model_global_edge: bool = False,
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
        self.property_prediction = property_prediction
        self.ligand_pocket_interaction = ligand_pocket_interaction
        self.store_intermediate_coords = store_intermediate_coords
        self.model_edge_rbf_interaction = model_edge_rbf_interaction
        self.model_global_edge = model_global_edge
        
        self.sdim, self.vdim = hn_dim
        self.edge_dim = edge_dim
        self.dynamic_graph = dynamic_graph
        convs = []
        
        self.cutoff_p = cutoff_local
        self.cutoff_lp = cutoff_local
        self.kNN = kNN

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
                    cutoff=cutoff_local,
                    use_rbfs=use_rbfs,
                    mask_pocket_edges=mask_pocket_edges,
                    model_edge_rbf_interaction = model_edge_rbf_interaction,
                    model_global_edge = model_global_edge,
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
        self.out_norm = LayerNorm(dims=hn_dim) if use_out_norm else None

        self.reset_parameters()

    def reset_parameters(self):
        for conv, norm in zip(self.convs, self.norms):
            conv.reset_parameters()
            norm.reset_parameters()
        if self.out_norm is not None:
            self.out_norm.reset_parameters()

    def calculate_edge_attrs(
        self,
        edge_index: Tensor,
        edge_attr: OptTensor,
        pos: Tensor,
        sqrt: bool = True,
        batch: Tensor = None,
    ):
        source, target = edge_index
        r = pos[target] - pos[source]

        if self.ligand_pocket_interaction:
            pos = pos / torch.norm(pos, dim=1).unsqueeze(1)
            a = pos[target] * pos[source]
        else:
            a = pos[target] * pos[source]
        a = a.sum(-1)
        d = torch.clamp(torch.pow(r, 2).sum(-1), min=1e-6)
        if sqrt:
            d = d.sqrt()
        r_norm = torch.div(r, (1.0 + d.unsqueeze(-1)))
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
        batch_lig: OptTensor = None,
        pocket_mask: OptTensor = None,
        edge_mask_pocket: OptTensor = None,
        edge_mask_ligand: OptTensor = None,
        batch_pocket: OptTensor = None,
        edge_attr_initial_ohe=None,
        edgt_attr_global_embedding=None,
    ) -> Dict:
        # edge_attr_xyz (distances, cosines, relative_positions, edge_features)
        # (E, E, E x 3, E x F)

        pos_list = []
        
        if self.dynamic_graph:
            E_dense = self.to_dense_edge_tensor(edge_index=edge_index_global,
                                                edge_attr=edge_attr_global[-1],
                                                num_nodes=s.size(0),
                                                )
        else:
            E_dense = 0.0
            
        for i in range(len(self.convs)):
            edge_index_in = edge_index_global
            edge_attr_in = edge_attr_global

            if context is not None and (i == 1 or i == len(self.convs) - 1):
                s = s + context
            s, v = self.norms[i](x={"s": s, "v": v, "z": z}, batch=batch)
            out = self.convs[i](
                x=(s, v, p),
                batch=batch,
                edge_index=edge_index_in,
                edge_attr=edge_attr_in,
                batch_lig=batch_lig,
                pocket_mask=pocket_mask,
                edge_mask_pocket=edge_mask_pocket,
                edge_mask_ligand=edge_mask_ligand,
                edge_attr_initial_ohe=edge_attr_initial_ohe,
                edgt_attr_global_embedding=edgt_attr_global_embedding,
            )

            s, v, p, e = out["s"], out["v"], out["p"], out["e"]
            
            if self.dynamic_graph:
                E_dense[edge_index_global[0], edge_index_global[1], :] = e
                with torch.no_grad():
                    pos_pocket = p[(~pocket_mask).squeeze()]
                    pos_ligand = p[pocket_mask.squeeze()]
                
                edge_index_global = get_edges(batch_mask_lig=batch_lig,
                                              batch_mask_pocket=batch_pocket,
                                              pos_lig=pos_ligand,
                                              pos_pocket=pos_pocket,
                                              cutoff_p=self.cutoff_p,
                                              cutoff_lp=self.cutoff_lp,
                                              return_full_adj=False,
                                              kNN=self.kNN,
                                              )
                edge_index_global = sort_edge_index(edge_index=edge_index_global,
                                                    sort_by_row=False)
                edge_index_global, _ = remove_self_loops(edge_index_global)
                edge_mask_ligand = (edge_index_global[0] < len(batch_lig)) & (edge_index_global[1] < len(batch_lig))
                edge_mask_pocket = (edge_index_global[0] >= len(batch_lig)) & (edge_index_global[1] >= len(batch_lig))
                e = E_dense[edge_index_global[0], edge_index_global[1], :]
                
            if self.recompute_edge_attributes:
                edge_attr_global = self.calculate_edge_attrs(
                    edge_index=edge_index_global,
                    pos=p,
                    edge_attr=e,
                    sqrt=True,
                    batch=batch if self.ligand_pocket_interaction else None,
                )
                
            if self.store_intermediate_coords and self.training:
                if i < len(self.convs) - 1:
                    if pocket_mask is not None:
                        pos_list.append(p[pocket_mask.squeeze(), :])
                    else:
                        pos_list.append(p)

                    p = p.detach()

            e = edge_attr_global[-1]

        if self.out_norm is not None:
            s, v = self.out_norm(x={"s": s, "v": v, "z": z}, batch=batch)
        out = {"s": s, "v": v, "e": e, "p": p, "p_list": pos_list,
               "edge_mask_ligand": edge_mask_ligand, "edge_mask_pocket": edge_mask_pocket
               }

        return out


class EQGATEdgeLocalGlobalGNN(nn.Module):
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
        ligand_pocket_interaction: bool = False,
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
        self.ligand_pocket_interaction = ligand_pocket_interaction

        if self.ligand_pocket_interaction:
            self.se3norm = SE3Norm()

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
        self,
        edge_index: Tensor,
        edge_attr: OptTensor,
        pos: Tensor,
        sqrt: bool = True,
        batch: Tensor = None,
    ):
        source, target = edge_index
        r = pos[target] - pos[source]
        if self.ligand_pocket_interaction:
            normed_pos = self.se3norm(pos, batch)
            a = normed_pos[target] * normed_pos[source]
        else:
            a = pos[target] * pos[source]
        a = a.sum(-1)
        d = torch.clamp(torch.pow(r, 2).sum(-1), min=1e-6)
        if sqrt:
            d = d.sqrt()
        r_norm = torch.div(r, (1.0 + d.unsqueeze(-1)))
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
        batch_lig: OptTensor = None,
        pocket_mask: OptTensor = None,
    ) -> Dict:
        # edge_attr_xyz (distances, cosines, relative_positions, edge_features)
        # (E, E, E x 3, E x F)

        for i in range(len(self.convs)):
            edge_index_in = edge_index_global
            edge_attr_in = edge_attr_global

            if context is not None and (i == 1 or i == len(self.convs) - 1):
                s = s + context
            s, v = self.norms[i](x={"s": s, "v": v, "z": z}, batch=batch)
            out = self.convs[i](
                x=(s, v, p),
                batch=batch,
                edge_index=edge_index_in,
                edge_attr=edge_attr_in,
                batch_lig=batch_lig,
                pocket_mask=pocket_mask,
            )

            s, v, p, e = out["s"], out["v"], out["p"], out["e"]
            # p = p - scatter_mean(p, batch, dim=0)[batch]
            if self.recompute_edge_attributes:
                edge_attr_global = self.calculate_edge_attrs(
                    edge_index=edge_index_global,
                    pos=p,
                    edge_attr=e,
                    sqrt=True,
                    batch=batch if self.ligand_pocket_interaction else None,
                )

            e = edge_attr_global[-1]

        out = {"s": s, "v": v, "e": e, "p": p}

        return out


class EQGATLocalGNN(nn.Module):
    """_summary_
    EQGAT GNN Network updating node-level scalar, vectors and potentially coordinates.
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
        intermediate_outs: bool = False,
        use_pos_norm: bool = False,
        use_out_norm: bool = True,
        coords_update: bool = False,
    ):
        super(EQGATLocalGNN, self).__init__()

        self.num_layers = num_layers
        self.cutoff_local = cutoff_local

        self.sdim, self.vdim = hn_dim
        self.edge_dim = edge_dim

        convs = []
        self.intermediate_outs = intermediate_outs
        self.coords_update = coords_update
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
                    use_pos_norm=use_pos_norm,
                    coords_update=coords_update,
                )
            )

        self.convs = nn.ModuleList(convs)
        self.norms = nn.ModuleList([LayerNorm(dims=hn_dim) for _ in range(num_layers)])

        self.out_norm = LayerNorm(dims=hn_dim) if use_out_norm else None

        self.reset_parameters()

    def reset_parameters(self):
        for conv, norm in zip(self.convs, self.norms):
            conv.reset_parameters()
            norm.reset_parameters()
        if self.out_norm is not None:
            self.out_norm.reset_parameters()

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

        if self.intermediate_outs:
            results = []
        else:
            results = None

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
            s, v, p = out["s"], out["v"], out["p"]

            if self.intermediate_outs:
                results.append(s)

        if self.out_norm is not None:
            s, v = self.out_norm(x={"s": s, "v": v}, batch=batch)
        out = {"s": s, "v": v, "p": p if self.coords_update else None}

        if self.intermediate_outs:
            return out, results
        else:
            return out


class EQGATDynamicLocalEdge(nn.Module):
    """_summary_
    EQGAT GNN Network updating node-level scalar, vectors and coordinates on the node level
    and all fully-connected edges
    After each iteration, the radius graph is computed again.
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
        intermediate_outs: bool = False,
        use_pos_norm: bool = False,
        store_intermediate_coords: bool = False,
    ):
        super(EQGATDynamicLocalEdge, self).__init__()

        self.num_layers = num_layers
        self.cutoff_local = cutoff_local

        self.sdim, self.vdim = hn_dim
        self.edge_dim = edge_dim

        convs = []
        self.intermediate_outs = intermediate_outs
        self.store_intermediate_coords = store_intermediate_coords

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
                    use_pos_norm=use_pos_norm,
                    coords_update=True,
                )
            )

        self.convs = nn.ModuleList(convs)
        self.norms = nn.ModuleList([LayerNorm(dims=hn_dim) for _ in range(num_layers)])
        self.edge_pre = nn.ModuleList(
            [DenseLayer(hn_dim[0], edge_dim) for _ in range(num_layers)]
        )
        self.edge_post = nn.ModuleList(
            [DenseLayer(edge_dim, edge_dim) for _ in range(num_layers)]
        )

        self.reset_parameters()

    def reset_parameters(self):
        for conv, norm, a, b in zip(
            self.convs, self.norms, self.edge_pre, self.edge_post
        ):
            conv.reset_parameters()
            norm.reset_parameters()
            a.reset_parameters()
            b.reset_parameters()

    def calculate_edge_attrs(
        self, edge_index: Tensor, edge_attr: OptTensor, pos: Tensor, sqrt: bool = True
    ):
        source, target = edge_index
        r = pos[target] - pos[source]
        pos = pos / torch.norm(pos, dim=1).unsqueeze(1)
        a = pos[target] * pos[source]
        a = a.sum(-1)
        d = torch.clamp(torch.pow(r, 2).sum(-1), min=1e-6)
        if sqrt:
            d = d.sqrt()
        r_norm = torch.div(r, (1.0 + d.unsqueeze(-1)))
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
        edge_index_global: Tensor,
        edge_attr_global: Tuple[Tensor],
        batch: Tensor = None,
        batch_lig=None,
        pocket_mask=None,
    ) -> Dict:

        if self.intermediate_outs:
            results = []
        else:
            results = None

        pos_list = []
        n = s.size(0)
        E = self.to_dense_edge_tensor(
            edge_index=edge_index_global, edge_attr=edge_attr_global, num_nodes=n
        )
        edge_index_in = radius_graph(
            x=p, r=self.cutoff_local, batch=batch, max_num_neighbors=32
        )
        local_mask = torch.ones((edge_index_in.size(-1), 1), device=s.device)
        local_mask = self.to_dense_edge_tensor(
            edge_index=edge_index_in, edge_attr=local_mask, num_nodes=n
        )
        for i in range(len(self.convs)):
            edge_attr_in = self.from_dense_edge_tensor(edge_index=edge_index_in, E=E)
            s, v = self.norms[i](x={"s": s, "v": v}, batch=batch)
            edge_attr_in = self.calculate_edge_attrs(
                edge_index=edge_index_in, edge_attr=edge_attr_in, pos=p, sqrt=True
            )
            out = self.convs[i](
                x=(s, v, p),
                batch=batch,
                edge_index=edge_index_in,
                edge_attr=edge_attr_in,
                batch_lig=batch_lig,
                pocket_mask=pocket_mask,
            )
            edge_attr_in = edge_attr_in[-1]
            s, v, p = out["s"], out["v"], out["p"]
            if self.store_intermediate_coords and self.training:
                if i < len(self.convs) - 1:
                    if pocket_mask is not None:
                        pos_list.append(p[pocket_mask.squeeze(), :])
                    else:
                        pos_list.append(p)
                    p = p.detach()
            f = self.edge_pre[i](s)
            e = F.silu(f[edge_index_in[0]] + f[edge_index_in[1]])
            e = self.edge_post[i](edge_attr_in + e)
            EE = self.to_dense_edge_tensor(
                edge_index=edge_index_in, edge_attr=e, num_nodes=n
            )
            E = (1.0 - local_mask) * E + local_mask * EE
            if self.intermediate_outs:
                results.append(s)

        e = E[edge_index_global[0], edge_index_global[1], :]

        out = {"s": s, "v": v, "e": e, "p": p, "p_list": pos_list}

        if self.intermediate_outs:
            return out, results
        else:
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


### MixGNN
class MixGNN(nn.Module):
    def __init__(
        self,
        hn_dim: Tuple[int, int] = (256, 64),
        edge_dim: Optional[int] = 16,
        cutoff_local: float = 5.0,
        num_layers: int = 5,
        latent_dim: Optional[int] = None,
        rbf_dim: Optional[int] = None,
        vector_aggr: str = "mean",
        property_prediction: bool = False,
        store_intermediate_coords: bool = False,
    ):
        super(MixGNN, self).__init__()

        self.num_layers = num_layers
        self.cutoff_local = cutoff_local
        self.property_prediction = property_prediction
        self.store_intermediate_coords = store_intermediate_coords

        self.sdim, self.vdim = hn_dim
        self.edge_dim = edge_dim

        self.pre_conv = ConvLayer(
            in_dims=hn_dim,
            out_dims=hn_dim,
            edge_dim=edge_dim,
            has_v_in=False,
            use_mlp_update=True,
            vector_aggr=vector_aggr,
            cutoff=None,
            rbf_dim=None,
        )
        convs = []
        for _ in range(num_layers - 2):
            convs.append(
                ConvLayer(
                    in_dims=hn_dim,
                    out_dims=hn_dim,
                    edge_dim=edge_dim,
                    has_v_in=True,
                    use_mlp_update=True,
                    vector_aggr=vector_aggr,
                    cutoff=cutoff_local,
                    rbf_dim=rbf_dim,
                )
            )
        self.convs = nn.ModuleList(convs)
        self.post_conv = ConvLayer(
            in_dims=hn_dim,
            out_dims=hn_dim,
            edge_dim=edge_dim,
            has_v_in=False,
            use_mlp_update=True,
            vector_aggr=vector_aggr,
            cutoff=None,
            rbf_dim=None,
        )
        if latent_dim:
            norm_module = AdaptiveLayerNorm
        else:
            norm_module = LayerNorm

        self.norms = nn.ModuleList(
            [
                norm_module(dims=hn_dim, latent_dim=latent_dim)
                for _ in range(num_layers - 2)
            ]
        )
        self.post_norm = nn.ModuleList(
            [norm_module(dims=hn_dim, latent_dim=latent_dim) for _ in range(2)]
        )

        self.edge_pre = nn.Sequential(
            DenseLayer(hn_dim[0], hn_dim[0], activation=nn.SiLU()),
            DenseLayer(hn_dim[0], edge_dim),
        )
        self.edge_post = nn.Sequential(
            DenseLayer(edge_dim, edge_dim, activation=nn.SiLU()),
            DenseLayer(edge_dim, edge_dim),
        )

        self.reset_parameters()

    def reset_parameters(self):
        for conv, norm in zip(self.convs, self.norms):
            conv.reset_parameters()
            norm.reset_parameters()
        self.pre_conv.reset_parameters()
        self.post_conv.reset_parameters()
        self.post_norm[0].reset_parameters()
        self.post_norm[1].reset_parameters()
        reset(self.edge_pre)
        reset(self.edge_post)

    def calculate_edge_attrs(
        self,
        edge_index: Tensor,
        edge_attr: OptTensor,
        pos: Tensor,
        sqrt: bool = True,
    ):
        source, target = edge_index
        r = pos[target] - pos[source]
        pos = pos / torch.norm(pos, dim=1).unsqueeze(1)
        a = pos[target] * pos[source]
        a = a.sum(-1)
        d = torch.clamp(torch.pow(r, 2).sum(-1), min=1e-6)
        if sqrt:
            d = d.sqrt()
        r_norm = torch.div(r, (1.0 + d.unsqueeze(-1)))
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
        edge_index_global: Tensor,
        edge_attr_global: Tensor,
        batch: Tensor,
        z: OptTensor = None,
        context: OptTensor = None,
        pocket_mask: OptTensor = None,
        **kwargs
    ) -> Dict:

        pos_list = []
        n = s.size(0)

        ## Fully Connected Pre
        E = self.to_dense_edge_tensor(
            edge_index=edge_index_global, edge_attr=edge_attr_global, num_nodes=n
        )
        edge_attr_in = self.calculate_edge_attrs(
            edge_index=edge_index_global,
            edge_attr=edge_attr_global,
            pos=p,
            sqrt=True,
        )
        out = self.pre_conv(
            x=(s, v, p),
            edge_index=edge_index_global,
            edge_attr=edge_attr_in,
            pocket_mask=pocket_mask,
        )
        s, v, p = out["s"], out["v"], out["p"]
        if self.store_intermediate_coords and self.training:
            if pocket_mask is not None:
                pos_list.append(p[pocket_mask.squeeze(), :])
            else:
                pos_list.append(p)
            p = p.detach()

        ## Dynamic Local
        for i in range(len(self.convs)):
            edge_index_in = radius_graph(
                x=p, r=self.cutoff_local, batch=batch, max_num_neighbors=128
            )
            edge_attr_in = self.from_dense_edge_tensor(edge_index=edge_index_in, E=E)
            edge_attr_in = self.calculate_edge_attrs(
                edge_index=edge_index_in,
                edge_attr=edge_attr_in,
                pos=p,
                sqrt=True,
            )

            if context is not None and (i == 0 or i == len(self.convs) - 1):
                s = s + context
            s, v = self.norms[i](x={"s": s, "v": v, "z": z}, batch=batch)
            out = self.convs[i](
                x=(s, v, p),
                edge_index=edge_index_in,
                edge_attr=edge_attr_in,
                pocket_mask=pocket_mask,
            )
            s, v, p = out["s"], out["v"], out["p"]

            if self.store_intermediate_coords and self.training:
                if i < len(self.convs) - 1:
                    if pocket_mask is not None:
                        pos_list.append(p[pocket_mask.squeeze(), :])
                    else:
                        pos_list.append(p)
                    p = p.detach()

        ## Fully Connected Post
        s, v = self.post_norm[0](x={"s": s, "v": v, "z": z}, batch=batch)
        edge_attr_in = self.calculate_edge_attrs(
            edge_index=edge_index_global,
            edge_attr=edge_attr_global,
            pos=p,
            sqrt=True,
        )
        out = self.post_conv(
            x=(s, v, p),
            edge_index=edge_index_global,
            edge_attr=edge_attr_in,
            pocket_mask=pocket_mask,
        )
        s, v, p = out["s"], out["v"], out["p"]
        s, v = self.post_norm[1](x={"s": s, "v": v, "z": z}, batch=batch)

        e = self.edge_pre(s)
        e = e[edge_index_global[0]] + e[edge_index_global[1]]
        e = self.edge_post(edge_attr_global + e)
        if self.store_intermediate_coords and self.training:
            if pocket_mask is not None:
                pos_list.append(p[pocket_mask.squeeze(), :])
            else:
                pos_list.append(p)
            p = p.detach()

        out = {"s": s, "v": v, "e": e, "p": p, "p_list": pos_list}
        return out