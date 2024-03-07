from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.nn import radius_graph
from torch_geometric.nn.inits import reset
from torch_geometric.typing import OptTensor
from torch_geometric.utils import softmax
from torch_scatter import scatter_add, scatter_mean

from e3moldiffusion.gnn import (
    EQGATEdgeGNN,
    EQGATEnergyGNN,
    EQGATLocalGNN,
)
from e3moldiffusion.modules import (
    DenseLayer,
    GatedEquivBlock,
    HiddenEdgeDistanceMLP,
    PredictionHeadEdge,
    PropertyPredictionHead,
    PropertyPredictionMLP,
)


class DenoisingEdgeNetwork(nn.Module):
    """_summary_
    Denoising network that inputs:
        atom features, edge features, position features
    The network is tasked for data prediction, i.e. x0 parameterization as commonly known in the literature:
        atom features, edge features, position features
    Args:
        nn (_type_): _description_
    """

    def __init__(
        self,
        num_atom_features: int,
        num_bond_types: int = 5,
        hn_dim: Tuple[int, int] = (256, 64),
        edge_dim: int = 32,
        cutoff_local: float = 7.5,
        num_layers: int = 5,
        latent_dim: Optional[int] = None,
        use_cross_product: bool = False,
        fully_connected: bool = True,
        local_global_model: bool = False,
        recompute_radius_graph: bool = True,
        recompute_edge_attributes: bool = True,
        vector_aggr: str = "mean",
        atom_mapping: bool = True,
        bond_mapping: bool = True,
        edge_mp: bool = False,
        p1: bool = True,
        use_pos_norm: bool = True,
        use_out_norm: bool = True,
        context_mapping: bool = False,
        num_context_features: int = 0,
        coords_param: str = "data",
        ligand_pocket_interaction: bool = False,
        store_intermediate_coords: bool = False,
        distance_ligand_pocket: bool = False,
        property_prediction: bool = False,
        joint_property_prediction: bool = False,
        regression_property: list = None,
        bond_prediction: bool = False,
        dynamic_graph: bool = False,
        kNN: Optional[int] = None,
        use_rbfs: bool = False,
    ) -> None:
        super(DenoisingEdgeNetwork, self).__init__()

        self.property_prediction = property_prediction
        self.joint_property_prediction = joint_property_prediction
        self.regression_property = regression_property
        self.bond_prediction = bond_prediction
        self.num_bond_types = num_bond_types

        self.ligand_pocket_interaction = ligand_pocket_interaction
        self.store_intermediate_coords = store_intermediate_coords

        self.time_mapping_atom = DenseLayer(1, hn_dim[0])
        self.time_mapping_bond = DenseLayer(1, edge_dim)

        if atom_mapping:
            self.atom_mapping = DenseLayer(num_atom_features, hn_dim[0])
        else:
            self.atom_mapping = nn.Identity()

        if bond_mapping or bond_prediction:
            if bond_prediction:
                num_bond_types = 1 * num_atom_features + 1
            self.bond_mapping = DenseLayer(num_bond_types, edge_dim)
        else:
            self.bond_mapping = nn.Identity()

        self.atom_time_mapping = DenseLayer(hn_dim[0], hn_dim[0])
        self.bond_time_mapping = DenseLayer(edge_dim, edge_dim)

        if context_mapping and latent_dim is None:
            self.context_mapping = True
            self.context_mapping = DenseLayer(num_context_features, hn_dim[0])
            self.atom_context_mapping = DenseLayer(hn_dim[0], hn_dim[0])

        else:
            self.context_mapping = False

        assert fully_connected or local_global_model

        self.sdim, self.vdim = hn_dim

        self.local_global_model = local_global_model
        self.fully_connected = fully_connected

        assert fully_connected
        assert not local_global_model

        self.gnn = EQGATEdgeGNN(
            hn_dim=hn_dim,
            cutoff_local=cutoff_local,
            num_atom_features=num_atom_features,
            num_bond_types=num_bond_types,
            coords_param=coords_param,
            num_context_features=num_context_features,
            property_prediction=property_prediction,
            edge_dim=edge_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
            use_cross_product=use_cross_product,
            vector_aggr=vector_aggr,
            fully_connected=fully_connected,
            local_global_model=local_global_model,
            recompute_radius_graph=recompute_radius_graph,
            recompute_edge_attributes=recompute_edge_attributes,
            edge_mp=edge_mp,
            p1=p1,
            use_pos_norm=use_pos_norm,
            use_out_norm=use_out_norm,
            ligand_pocket_interaction=ligand_pocket_interaction,
            store_intermediate_coords=store_intermediate_coords,
            dynamic_graph=dynamic_graph,
            kNN=kNN,
            use_rbfs=use_rbfs,
        )

        if property_prediction:
            self.prediction_head = PropertyPredictionHead(
                hn_dim=hn_dim,
                num_context_features=num_context_features,
            )
        else:
            self.prediction_head = PredictionHeadEdge(
                hn_dim=hn_dim,
                edge_dim=edge_dim,
                num_atom_features=num_atom_features,
                num_bond_types=num_bond_types,
                coords_param=coords_param,
                joint_property_prediction=self.joint_property_prediction,
                regression_property=self.regression_property,
            )
            # self.prediction_head = PredictionHeadEdge_Old(
            #     hn_dim=hn_dim,
            #     edge_dim=edge_dim,
            #     num_atom_features=num_atom_features,
            #     num_bond_types=num_bond_types,
            #     coords_param=coords_param,
            #     joint_property_prediction=self.joint_property_prediction,
            # )

        self.distance_ligand_pocket = distance_ligand_pocket
        if distance_ligand_pocket:
            self.ligand_pocket_mlp = HiddenEdgeDistanceMLP(hn_dim=hn_dim)
        else:
            self.ligand_pocket_mlp = None

        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self.atom_mapping, "reset_parameters"):
            self.atom_mapping.reset_parameters()
        if hasattr(self.bond_mapping, "reset_parameters"):
            self.bond_mapping.reset_parameters()
        self.time_mapping_atom.reset_parameters()
        self.atom_time_mapping.reset_parameters()
        if self.context_mapping and hasattr(
            self.atom_context_mapping, "reset_parameters"
        ):
            self.atom_context_mapping.reset_parameters()
        if self.context_mapping and hasattr(self.context_mapping, "reset_parameters"):
            self.context_mapping.reset_parameters()
        self.time_mapping_bond.reset_parameters()
        self.bond_time_mapping.reset_parameters()
        self.gnn.reset_parameters()

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

    def forward(
        self,
        x: Tensor,
        t: Tensor,
        pos: Tensor,
        edge_index_local: Tensor,
        edge_index_global: Tensor,
        edge_index_global_lig: OptTensor = None,
        edge_attr_global: OptTensor = None,
        batch: OptTensor = None,
        batch_edge_global: OptTensor = None,
        z: OptTensor = None,
        context: OptTensor = None,
        pocket_mask: OptTensor = None,
        edge_mask: OptTensor = None,
        edge_mask_pocket: OptTensor = None,
        ca_mask: OptTensor = None,
        batch_pocket: OptTensor = None,
        pos_lig: OptTensor = None,
        atoms_lig: OptTensor = None,
        edge_attr_global_lig: OptTensor = None,
        batch_edge_global_lig: OptTensor = None,
        batch_lig: OptTensor = None,
        joint_tensor: OptTensor = None,
    ) -> Dict:

        if pos is None and x is None:
            assert joint_tensor is not None
            pos = joint_tensor[:, :3].clone()
            x = joint_tensor[:, 3:].clone()

        if pocket_mask is None:
            pos = pos - scatter_mean(pos, index=batch, dim=0)[batch]
        # t: (batch_size,)
        ta = self.time_mapping_atom(t)
        tb = self.time_mapping_bond(t)
        tnode = ta[batch]

        # edge_index_global (2, E*)
        tedge_global = tb[batch_edge_global]

        if batch is None:
            batch = torch.zeros(x.size(0), device=x.device, dtype=torch.long)

        s = self.atom_mapping(x)
        cemb = None
        if context is not None and self.context_mapping:
            cemb = self.context_mapping(context)
            s = self.atom_context_mapping(s + cemb)
        s = self.atom_time_mapping(s + tnode)

        if self.bond_prediction:
            # symmetric initial edge-feature
            d = (
                (pos[edge_index_global[1]] - pos[edge_index_global[0]])
                .pow(2)
                .sum(-1, keepdim=True)
                .sqrt()
            )
            edge_attr_global = torch.concat(
                [x[edge_index_global[1]] + x[edge_index_global[0]], d], dim=-1
            )
        edge_attr_global_transformed = self.bond_mapping(edge_attr_global)
        edge_attr_global_transformed = self.bond_time_mapping(
            edge_attr_global_transformed + tedge_global
        )

        # edge_dense = torch.zeros(x.size(0), x.size(0), edge_attr_global_transformed.size(-1), device=s.device)
        # edge_dense[edge_index_global[0], edge_index_global[1], :] = edge_attr_global_transformed

        # if not self.fully_connected:
        #    edge_attr_local_transformed = edge_dense[edge_index_local[0], edge_index_local[1], :]
        #    # local
        #    edge_attr_local_transformed = self.calculate_edge_attrs(edge_index=edge_index_local, edge_attr=edge_attr_local_transformed, pos=pos)
        # else:
        #    edge_attr_local_transformed = (None, None, None)

        v = torch.zeros(size=(x.size(0), 3, self.vdim), device=s.device)

        # global
        edge_attr_global_transformed = self.calculate_edge_attrs(
            edge_index=edge_index_global,
            edge_attr=edge_attr_global_transformed,
            pos=pos,
            sqrt=True,
            batch=batch if self.ligand_pocket_interaction else None,
        )

        out = self.gnn(
            s=s,
            v=v,
            p=pos,
            z=z,
            edge_index_local=None,
            edge_attr_local=(None, None, None),
            edge_index_global=edge_index_global,
            edge_attr_global=edge_attr_global_transformed,
            batch=batch,
            context=cemb,
            batch_lig=batch_lig,
            pocket_mask=pocket_mask,
            edge_mask_pocket=edge_mask_pocket,
            edge_mask_ligand=edge_mask,
            batch_pocket=batch_pocket,
        )
        edge_mask, edge_mask_pocket = out["edge_mask_ligand"], out["edge_mask_pocket"]

        coords_pred, atoms_pred, bonds_pred, property_pred = self.prediction_head(
            x=out,
            batch=batch,
            edge_index_global=edge_index_global,
            edge_index_global_lig=edge_index_global_lig,
            batch_lig=batch_lig,
            pocket_mask=pocket_mask,
            edge_mask=edge_mask,
        )

        if self.store_intermediate_coords and self.training:
            pos_list = out["p_list"]
            assert len(pos_list) > 0
            pos_list.append(coords_pred)
            coords_pred = torch.stack(pos_list, dim=0)  # [num_layers, N, 3]

        if self.distance_ligand_pocket:
            dist_pred = self.ligand_pocket_mlp(
                x=out,
                batch_ligand=batch_lig,
                batch_pocket=batch_pocket,
                pocket_mask=pocket_mask,
                ca_mask=ca_mask,
            )
        else:
            dist_pred = None

        out = {
            "coords_pred": coords_pred,
            "atoms_pred": atoms_pred,
            "bonds_pred": bonds_pred,
            "dist_pred": dist_pred,
            "property_pred": property_pred,
        }

        return out


class LatentEncoderNetwork(nn.Module):
    def __init__(
        self,
        num_atom_features: int,
        num_bond_types: int = 5,
        hn_dim: Tuple[int, int] = (256, 64),
        edge_dim: int = 32,
        cutoff_local: float = 7.5,
        num_layers: int = 5,
        use_cross_product: bool = False,
        vector_aggr: str = "mean",
        atom_mapping: bool = True,
        bond_mapping: bool = True,
        intermediate_outs: bool = False,
        use_pos_norm: bool = False,
        use_out_norm: bool = False,
    ) -> None:
        super(LatentEncoderNetwork, self).__init__()

        if atom_mapping:
            self.atom_mapping = DenseLayer(num_atom_features, hn_dim[0])
        else:
            self.atom_mapping = nn.Identity()

        if bond_mapping:
            self.bond_mapping = DenseLayer(num_bond_types, edge_dim)
        else:
            self.bond_mapping = nn.Identity()

        self.sdim, self.vdim = hn_dim

        self.gnn = EQGATLocalGNN(
            hn_dim=hn_dim,
            cutoff_local=cutoff_local,
            edge_dim=edge_dim,
            num_layers=num_layers,
            use_cross_product=use_cross_product,
            vector_aggr=vector_aggr,
            intermediate_outs=intermediate_outs,
            use_pos_norm=use_pos_norm,
            use_out_norm=use_out_norm,
            coords_update=False,
        )

        self.intermediate_outs = intermediate_outs

        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self.atom_mapping, "reset_parameters"):
            self.atom_mapping.reset_parameters()
        if hasattr(self.bond_mapping, "reset_parameters"):
            self.bond_mapping.reset_parameters()
        self.gnn.reset_parameters()

    def calculate_edge_attrs(
        self, edge_index: Tensor, edge_attr: OptTensor, pos: Tensor, sqrt: bool = True
    ):
        source, target = edge_index
        r = pos[target] - pos[source]
        pos_norm = F.normalize(pos, p=2, dim=-1)
        a = pos_norm[target] * pos_norm[source]
        a = a.sum(-1)
        d = torch.clamp(torch.pow(r, 2).sum(-1), min=1e-6)
        if sqrt:
            d = d.sqrt()
        r_norm = torch.div(r, (d.unsqueeze(-1) + 1.0))
        edge_attr = (d, a, r_norm, edge_attr)
        return edge_attr

    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        edge_index_local: Tensor,
        edge_attr_local: OptTensor = Tensor,
        batch: OptTensor = None,
    ) -> Dict:
        pos = pos - scatter_mean(pos, index=batch, dim=0)[batch]

        if batch is None:
            batch = torch.zeros(x.size(0), device=x.device, dtype=torch.long)

        s = self.atom_mapping(x)

        edge_attr_local_transformed = self.bond_mapping(edge_attr_local)
        edge_attr_local_transformed = self.calculate_edge_attrs(
            edge_index=edge_index_local,
            edge_attr=edge_attr_local_transformed,
            pos=pos,
            sqrt=True,
        )

        v = torch.zeros(size=(x.size(0), 3, self.vdim), device=s.device)

        if not self.intermediate_outs:
            out = self.gnn(
                s=s,
                v=v,
                p=pos,
                edge_index_local=edge_index_local,
                edge_attr_local=edge_attr_local_transformed,
                edge_index_global=None,
                edge_attr_global=None,
                batch=batch,
            )
            return out
        else:
            out, scalars = self.gnn(
                s=s,
                v=v,
                p=pos,
                edge_index_local=edge_index_local,
                edge_attr_local=edge_attr_local_transformed,
                edge_index_global=None,
                edge_attr_global=None,
                batch=batch,
            )
            return out, scalars


class SoftMaxAttentionAggregation(nn.Module):
    """
    Softmax Attention Pooling as proposed "Graph Matching Networks
    for Learning the Similarity of Graph Structured Objects"
    <https://arxiv.org/abs/1904.12787>
    """

    def __init__(self, dim: int):
        super(SoftMaxAttentionAggregation, self).__init__()

        self.node_net = nn.Sequential(
            DenseLayer(dim, dim, activation=nn.SiLU()), DenseLayer(dim, dim)
        )
        self.gate_net = nn.Sequential(
            DenseLayer(dim, dim, activation=nn.SiLU()), DenseLayer(dim, 1)
        )
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.node_net)
        reset(self.gate_net)

    def forward(
        self,
        x: Tensor,
        index: Optional[Tensor] = None,
        dim_size: Optional[int] = None,
        dim: int = -2,
    ) -> Tensor:
        if index is None:
            index = torch.zeros(size=(x.size(0)), device=x.device, dtype=torch.long)
        if dim_size is None:
            dim_size = int(index.max()) + 1
        gate = self.gate_net(x)

        gate = softmax(gate, index, dim=0)
        x = self.node_net(x)
        x = gate * x
        x = scatter_add(src=x, index=index, dim=dim, dim_size=dim_size)
        return x


class EdgePredictionHead(nn.Module):
    def __init__(
        self,
        hn_dim: Tuple[int, int],
        edge_dim: int,
        num_atom_features: int,
        num_bond_types: int = 5,
    ) -> None:
        super(EdgePredictionHead, self).__init__()
        self.sdim, self.vdim = hn_dim
        self.num_atom_features = num_atom_features

        self.shared_mapping = DenseLayer(
            self.sdim, self.sdim, bias=True, activation=nn.SiLU()
        )

        self.bond_mapping = DenseLayer(edge_dim, self.sdim, bias=True)

        self.bonds_lin_0 = DenseLayer(
            in_features=self.sdim + 1, out_features=self.sdim, bias=True
        )
        self.bonds_lin_1 = DenseLayer(
            in_features=self.sdim, out_features=num_bond_types, bias=True
        )
        self.coords_lin = DenseLayer(in_features=self.vdim, out_features=1, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.shared_mapping.reset_parameters()
        self.coords_lin.reset_parameters()
        self.bonds_lin_0.reset_parameters()
        self.bonds_lin_1.reset_parameters()

    def forward(self, x: Dict, batch: Tensor, edge_index_global: Tensor) -> Dict:
        s, v, p, e = x["s"], x["v"], x["p"], x["e"]
        s = self.shared_mapping(s)
        j, i = edge_index_global
        n = s.size(0)

        coords_pred = self.coords_lin(v).squeeze()

        coords_pred = p + coords_pred
        coords_pred = coords_pred - scatter_mean(coords_pred, index=batch, dim=0)[batch]

        e_dense = torch.zeros(n, n, e.size(-1), device=e.device)
        e_dense[edge_index_global[0], edge_index_global[1], :] = e
        e_dense = 0.5 * (e_dense + e_dense.permute(1, 0, 2))
        e = e_dense[edge_index_global[0], edge_index_global[1], :]

        d = (coords_pred[i] - coords_pred[j]).pow(2).sum(-1, keepdim=True)  # .sqrt()
        f = s[i] + s[j] + self.bond_mapping(e)
        edge = torch.cat([f, d], dim=-1)

        bonds_pred = F.silu(self.bonds_lin_0(edge))
        bonds_pred = self.bonds_lin_1(bonds_pred)

        return bonds_pred


class EdgePredictionNetwork(nn.Module):
    """_summary_
    Denoising network that inputs:
        atom features, edge features, position features
    The network is tasked for data prediction, i.e. x0 parameterization as commonly known in the literature:
        atom features, edge features, position features
    Args:
        nn (_type_): _description_
    """

    def __init__(
        self,
        num_atom_features: int,
        num_bond_types: int = 5,
        hn_dim: Tuple[int, int] = (256, 64),
        edge_dim: int = 32,
        cutoff_local: float = 7.5,
        num_layers: int = 5,
        latent_dim: Optional[int] = None,
        use_cross_product: bool = False,
        fully_connected: bool = True,
        local_global_model: bool = False,
        recompute_radius_graph: bool = True,
        recompute_edge_attributes: bool = True,
        vector_aggr: str = "mean",
        atom_mapping: bool = True,
        bond_mapping: bool = True,
        edge_mp: bool = False,
        p1: bool = True,
        use_pos_norm: bool = True,
    ) -> None:
        super(EdgePredictionNetwork, self).__init__()

        self.time_mapping_atom = DenseLayer(1, hn_dim[0])
        self.time_mapping_bond = DenseLayer(1, edge_dim)

        if atom_mapping:
            self.atom_mapping = DenseLayer(num_atom_features, hn_dim[0])
        else:
            self.atom_mapping = nn.Identity()

        if bond_mapping:
            self.bond_mapping = DenseLayer(
                num_atom_features, edge_dim
            )  # BOND PREDICTION
        else:
            self.bond_mapping = nn.Identity()

        self.atom_time_mapping = DenseLayer(hn_dim[0], hn_dim[0])
        self.bond_time_mapping = DenseLayer(edge_dim, edge_dim)

        assert fully_connected or local_global_model

        self.sdim, self.vdim = hn_dim

        self.local_global_model = local_global_model
        self.fully_connected = fully_connected

        assert fully_connected
        assert not local_global_model

        self.gnn = EQGATEdgeGNN(
            hn_dim=hn_dim,
            cutoff_local=cutoff_local,
            edge_dim=edge_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
            use_cross_product=use_cross_product,
            vector_aggr=vector_aggr,
            fully_connected=fully_connected,
            local_global_model=local_global_model,
            recompute_radius_graph=recompute_radius_graph,
            recompute_edge_attributes=recompute_edge_attributes,
            edge_mp=edge_mp,
            p1=p1,
            use_pos_norm=use_pos_norm,
        )

        self.prediction_head = EdgePredictionHead(
            hn_dim=hn_dim,
            edge_dim=edge_dim,
            num_atom_features=num_atom_features,
            num_bond_types=num_bond_types,
        )

        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self.atom_mapping, "reset_parameters"):
            self.atom_mapping.reset_parameters()
        if hasattr(self.bond_mapping, "reset_parameters"):
            self.bond_mapping.reset_parameters()
        self.time_mapping_atom.reset_parameters()
        self.atom_time_mapping.reset_parameters()
        self.time_mapping_bond.reset_parameters()
        self.bond_time_mapping.reset_parameters()
        self.gnn.reset_parameters()
        self.prediction_head.reset_parameters()

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
        r_norm = torch.div(r, (1.0 + d.unsqueeze(-1)))
        edge_attr = (d, a, r_norm, edge_attr)
        return edge_attr

    def forward(
        self,
        x: Tensor,
        t: Tensor,
        pos: Tensor,
        edge_index_local: Tensor,
        edge_index_global: Tensor,
        edge_attr_global: OptTensor = Tensor,
        batch: OptTensor = None,
        batch_edge_global: OptTensor = None,
        z: OptTensor = None,
    ) -> Dict:
        pos = pos - scatter_mean(pos, index=batch, dim=0)[batch]
        # t: (batch_size,)
        ta = self.time_mapping_atom(t)
        tb = self.time_mapping_bond(t)
        tnode = ta[batch]

        # edge_index_global (2, E*)
        tedge_global = tb[batch_edge_global]

        if batch is None:
            batch = torch.zeros(x.size(0), device=x.device, dtype=torch.long)

        s = self.atom_mapping(x)
        s = self.atom_time_mapping(s + tnode)

        edge_attr_global = x[edge_index_global[1]]

        edge_attr_global_transformed = self.bond_mapping(edge_attr_global)
        edge_attr_global_transformed = self.bond_time_mapping(
            edge_attr_global_transformed + tedge_global
        )

        # global
        edge_attr_global_transformed = self.calculate_edge_attrs(
            edge_index=edge_index_global,
            edge_attr=edge_attr_global_transformed,
            pos=pos,
            sqrt=True,
        )

        v = torch.zeros(size=(x.size(0), 3, self.vdim), device=s.device)

        out = self.gnn(
            s=s,
            v=v,
            p=pos,
            z=z,
            edge_index_local=None,
            edge_attr_local=(None, None, None),
            edge_index_global=edge_index_global,
            edge_attr_global=edge_attr_global_transformed,
            batch=batch,
        )

        out = self.prediction_head(
            x=out, batch=batch, edge_index_global=edge_index_global
        )

        return out


class EQGATEnergyNetwork(nn.Module):
    def __init__(
        self,
        num_atom_features: int,
        hn_dim: Tuple[int, int] = (256, 64),
        num_rbfs: int = 20,
        cutoff_local: float = 5.0,
        num_layers: int = 5,
        use_cross_product: bool = False,
        vector_aggr: str = "mean",
    ) -> None:
        super().__init__()
        self.cutoff = cutoff_local
        self.sdim, self.vdim = hn_dim

        self.time_mapping_atom = DenseLayer(1, hn_dim[0])
        self.atom_mapping = DenseLayer(num_atom_features, hn_dim[0])
        self.atom_time_mapping = DenseLayer(hn_dim[0], hn_dim[0])

        self.gnn = EQGATEnergyGNN(
            hn_dim=hn_dim,
            cutoff=cutoff_local,
            num_rbfs=num_rbfs,
            num_layers=num_layers,
            use_cross_product=use_cross_product,
            vector_aggr=vector_aggr,
        )
        self.energy_head = GatedEquivBlock(
            in_dims=hn_dim, out_dims=(1, None), use_mlp=True
        )

    def calculate_edge_attrs(self, edge_index: Tensor, pos: Tensor):
        source, target = edge_index
        r = pos[target] - pos[source]
        d = torch.clamp(torch.pow(r, 2).sum(-1), min=1e-6)
        d = d.sqrt()
        # r_norm = torch.div(r, (1.0 + d.unsqueeze(-1)))
        r_norm = torch.div(r, (d.unsqueeze(-1)))
        edge_attr = (d, r_norm)
        return edge_attr

    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        t: Tensor,
        batch: OptTensor = None,
        joint_tensor: OptTensor = None,
    ) -> Dict:
        if pos is None and x is None:
            assert joint_tensor is not None
            pos = joint_tensor[:, :3].clone()
            x = joint_tensor[:, 3:].clone()

        edge_index = radius_graph(
            x=pos, r=self.cutoff, batch=batch, max_num_neighbors=128
        )
        ta = self.time_mapping_atom(t)
        tnode = ta[batch]
        s = self.atom_mapping(x)
        s = self.atom_time_mapping(s + tnode)
        v = torch.zeros(
            size=(x.size(0), 3, self.vdim), device=x.device, dtype=pos.dtype
        )

        edge_attr = self.calculate_edge_attrs(edge_index, pos)
        s, v = self.gnn(
            s=s, v=v, edge_index=edge_index, edge_attr=edge_attr, batch=batch
        )
        energy_atoms, v = self.energy_head((s, v))
        energy_atoms = energy_atoms + v.sum() * 0
        bs = len(batch.unique())
        energy_molecule = scatter_add(energy_atoms, index=batch, dim=0, dim_size=bs)
        assert energy_molecule.size(1) == 1
        out = {"property_pred": energy_molecule}
        return out


class PropertyEdgeNetwork(nn.Module):
    """_summary_
    Denoising network that inputs:
        atom features, edge features, position features
    The network is tasked for data prediction, i.e. x0 parameterization as commonly known in the literature:
        atom features, edge features, position features
    Args:
        nn (_type_): _description_
    """

    def __init__(
        self,
        num_atom_features: int,
        num_bond_types: int = 5,
        hn_dim: Tuple[int, int] = (256, 64),
        edge_dim: int = 32,
        cutoff_local: float = 7.5,
        num_layers: int = 5,
        latent_dim: Optional[int] = None,
        use_cross_product: bool = False,
        fully_connected: bool = True,
        local_global_model: bool = False,
        recompute_radius_graph: bool = True,
        recompute_edge_attributes: bool = True,
        vector_aggr: str = "mean",
        atom_mapping: bool = True,
        bond_mapping: bool = True,
        edge_mp: bool = False,
        p1: bool = True,
        use_pos_norm: bool = True,
        context_mapping: bool = False,
        num_context_features: int = 0,
        coords_param: str = "data",
        ligand_pocket_interaction: bool = False,
        store_intermediate_coords: bool = False,
        property_prediction: bool = False,
        joint_property_prediction: bool = False,
        bond_prediction: bool = False,
    ) -> None:
        super(PropertyEdgeNetwork, self).__init__()

        self.property_prediction = property_prediction
        self.joint_property_prediction = joint_property_prediction
        self.bond_prediction = bond_prediction
        self.num_bond_types = num_bond_types

        self.ligand_pocket_interaction = ligand_pocket_interaction
        self.store_intermediate_coords = store_intermediate_coords

        self.time_mapping_atom = DenseLayer(1, hn_dim[0])
        self.time_mapping_bond = DenseLayer(1, edge_dim)

        if atom_mapping:
            self.atom_mapping = DenseLayer(num_atom_features, hn_dim[0])
        else:
            self.atom_mapping = nn.Identity()

        if bond_mapping or bond_prediction:
            if bond_prediction:
                num_bond_types = 1 * num_atom_features + 1
            self.bond_mapping = DenseLayer(num_bond_types, edge_dim)
        else:
            self.bond_mapping = nn.Identity()

        self.atom_time_mapping = DenseLayer(hn_dim[0], hn_dim[0])
        self.bond_time_mapping = DenseLayer(edge_dim, edge_dim)

        self.context_mapping = context_mapping
        if self.context_mapping:
            self.context_mapping = DenseLayer(num_context_features, hn_dim[0])
            self.atom_context_mapping = DenseLayer(hn_dim[0], hn_dim[0])

        assert fully_connected or local_global_model

        self.sdim, self.vdim = hn_dim

        self.local_global_model = local_global_model
        self.fully_connected = fully_connected

        assert fully_connected
        assert not local_global_model

        if latent_dim:
            if context_mapping:
                latent_dim_ = None
            else:
                latent_dim_ = latent_dim
        else:
            latent_dim_ = None

        self.gnn = EQGATEdgeGNN(
            hn_dim=hn_dim,
            cutoff_local=cutoff_local,
            num_atom_features=num_atom_features,
            num_bond_types=num_bond_types,
            coords_param=coords_param,
            num_context_features=num_context_features,
            property_prediction=property_prediction,
            edge_dim=edge_dim,
            latent_dim=latent_dim_,
            num_layers=num_layers,
            use_cross_product=use_cross_product,
            vector_aggr=vector_aggr,
            fully_connected=fully_connected,
            local_global_model=local_global_model,
            recompute_radius_graph=recompute_radius_graph,
            recompute_edge_attributes=recompute_edge_attributes,
            edge_mp=edge_mp,
            p1=p1,
            use_pos_norm=use_pos_norm,
            ligand_pocket_interaction=ligand_pocket_interaction,
            store_intermediate_coords=store_intermediate_coords,
        )

        self.prediction_head = PropertyPredictionMLP(
            hn_dim=hn_dim,
            edge_dim=edge_dim,
            num_context_features=num_context_features,
            activation=nn.Identity(),
        )

        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self.atom_mapping, "reset_parameters"):
            self.atom_mapping.reset_parameters()
        if hasattr(self.bond_mapping, "reset_parameters"):
            self.bond_mapping.reset_parameters()
        self.time_mapping_atom.reset_parameters()
        self.atom_time_mapping.reset_parameters()
        if self.context_mapping and hasattr(
            self.atom_context_mapping, "reset_parameters"
        ):
            self.atom_context_mapping.reset_parameters()
        if self.context_mapping and hasattr(self.context_mapping, "reset_parameters"):
            self.context_mapping.reset_parameters()
        self.time_mapping_bond.reset_parameters()
        self.bond_time_mapping.reset_parameters()
        self.gnn.reset_parameters()

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

    def forward(
        self,
        x: Tensor,
        t: Tensor,
        pos: Tensor,
        edge_index_local: Tensor,
        edge_index_global: Tensor,
        edge_index_global_lig: OptTensor = None,
        edge_attr_global: OptTensor = None,
        batch: OptTensor = None,
        batch_edge_global: OptTensor = None,
        z: OptTensor = None,
        context: OptTensor = None,
        pocket_mask: OptTensor = None,
        edge_mask: OptTensor = None,
        edge_mask_pocket: OptTensor = None,
        ca_mask: OptTensor = None,
        batch_pocket: OptTensor = None,
        pos_lig: OptTensor = None,
        atoms_lig: OptTensor = None,
        edge_attr_global_lig: OptTensor = None,
        batch_edge_global_lig: OptTensor = None,
        batch_lig: OptTensor = None,
        joint_tensor: OptTensor = None,
    ) -> Dict:

        if pos is None and x is None:
            assert joint_tensor is not None
            pos = joint_tensor[:, :3].clone()
            x = joint_tensor[:, 3:].clone()

        if pocket_mask is None:
            pos = pos - scatter_mean(pos, index=batch, dim=0)[batch]
        # t: (batch_size,)
        ta = self.time_mapping_atom(t)
        tb = self.time_mapping_bond(t)
        tnode = ta[batch]

        # edge_index_global (2, E*)
        tedge_global = tb[batch_edge_global]

        if batch is None:
            batch = torch.zeros(x.size(0), device=x.device, dtype=torch.long)

        s = self.atom_mapping(x)
        cemb = None
        if context is not None and self.context_mapping:
            cemb = self.context_mapping(context)
            s = self.atom_context_mapping(s + cemb)
        s = self.atom_time_mapping(s + tnode)

        if self.bond_prediction:
            # symmetric initial edge-feature
            d = (
                (pos[edge_index_global[1]] - pos[edge_index_global[0]])
                .pow(2)
                .sum(-1, keepdim=True)
                .sqrt()
            )
            edge_attr_global = torch.concat(
                [x[edge_index_global[1]] + x[edge_index_global[0]], d], dim=-1
            )
        edge_attr_global_transformed = self.bond_mapping(edge_attr_global)
        edge_attr_global_transformed = self.bond_time_mapping(
            edge_attr_global_transformed + tedge_global
        )

        v = torch.zeros(size=(x.size(0), 3, self.vdim), device=s.device)

        # global
        edge_attr_global_transformed = self.calculate_edge_attrs(
            edge_index=edge_index_global,
            edge_attr=edge_attr_global_transformed,
            pos=pos,
            sqrt=True,
            batch=batch if self.ligand_pocket_interaction else None,
        )

        out = self.gnn(
            s=s,
            v=v,
            p=pos,
            z=z,
            edge_index_local=None,
            edge_attr_local=(None, None, None),
            edge_index_global=edge_index_global,
            edge_attr_global=edge_attr_global_transformed,
            batch=batch,
            context=cemb,
            batch_lig=batch_lig,
            pocket_mask=pocket_mask,
            edge_mask_pocket=edge_mask_pocket,
        )

        property_mol = self.prediction_head(out, batch, edge_index_global)

        assert property_mol.size(1) == 1

        out = {"property_pred": property_mol}
        return out


if __name__ == "__main__":
    pass
