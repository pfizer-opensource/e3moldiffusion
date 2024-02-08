import math
from typing import Callable, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.init import kaiming_uniform_, zeros_
from torch_geometric.nn.inits import reset
from torch_scatter import scatter_add, scatter_mean


class PredictionHeadEdge(nn.Module):
    def __init__(
        self,
        hn_dim: Tuple[int, int],
        edge_dim: int,
        num_atom_features: int,
        num_bond_types: int = 5,
        coords_param: str = "data",
        joint_property_prediction: bool = False,
    ) -> None:
        super(PredictionHeadEdge, self).__init__()
        self.sdim, self.vdim = hn_dim
        self.num_atom_features = num_atom_features
        self.coords_param = coords_param
        self.joint_property_prediction = joint_property_prediction

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
        self.atoms_lin = DenseLayer(
            in_features=self.sdim, out_features=num_atom_features, bias=True
        )
        if self.joint_property_prediction:
            #self.property_mlp = DenseLayer(
            #    in_features=self.sdim, out_features=1, bias=True, activation=nn.ReLU()
            #)
            self.property_mlp = nn.Sequential(DenseLayer(in_features=self.sdim,
                                                         out_features=self.sdim,
                                                         bias=True,
                                                         activation=nn.SiLU()
                                                         ),
                                              DenseLayer(in_features=self.sdim,
                                                         out_features=1,
                                                         bias=True,
                                                         activation=nn.Identity()
                                                         )
            )
        self.reset_parameters()

    def reset_parameters(self):
        self.shared_mapping.reset_parameters()
        self.coords_lin.reset_parameters()
        self.atoms_lin.reset_parameters()
        self.bonds_lin_0.reset_parameters()
        self.bonds_lin_1.reset_parameters()
        if self.joint_property_prediction:
            reset(self.property_mlp)

    def forward(
        self,
        x: Dict,
        batch: Tensor,
        edge_index_global: Tensor,
        edge_index_global_lig: Tensor = None,
        batch_lig: Tensor = None,
        pocket_mask: Tensor = None,
        edge_mask: Tensor = None,
    ) -> Dict:
        
        s, v, p, e = x["s"], x["v"], x["p"], x["e"]
        s = self.shared_mapping(s)
        coords_pred = self.coords_lin(v).squeeze()
        atoms_pred = self.atoms_lin(s)

        if self.joint_property_prediction:
            batch_size = len(batch.bincount())
            property_pred = self.property_mlp(s)
            property_pred = scatter_add(
                property_pred, index=batch, dim=0, dim_size=batch_size
            )
        else:
            property_pred = None

        if batch_lig is not None and pocket_mask is not None:
            s = (s * pocket_mask)[pocket_mask.squeeze(), :]
            j, i = edge_index_global_lig
            atoms_pred = (atoms_pred * pocket_mask)[pocket_mask.squeeze(), :]
            coords_pred = (coords_pred * pocket_mask)[pocket_mask.squeeze(), :]
            p = (p * pocket_mask)[pocket_mask.squeeze(), :]
            coords_pred = p + coords_pred
            d = (coords_pred[i] - coords_pred[j]).pow(2).sum(-1, keepdim=True)
        elif self.coords_param == "data":
            j, i = edge_index_global
            n = s.size(0)
            coords_pred = p + coords_pred
            coords_pred = (
                coords_pred - scatter_mean(coords_pred, index=batch, dim=0)[batch]
            )
            d = (
                (coords_pred[i] - coords_pred[j]).pow(2).sum(-1, keepdim=True)
            )  # .sqrt()
        else:
            j, i = edge_index_global
            n = s.size(0)
            d = (p[i] - p[j]).pow(2).sum(-1, keepdim=True)  # .sqrt()
            coords_pred = (
                coords_pred - scatter_mean(coords_pred, index=batch, dim=0)[batch]
            )

        if edge_mask is not None and edge_index_global_lig is not None:
            n = len(batch_lig)
            e = (e * edge_mask.unsqueeze(1))[edge_mask]
            e_dense = torch.zeros(n, n, e.size(-1), device=e.device)
            e_dense[edge_index_global_lig[0], edge_index_global_lig[1], :] = e
            e_dense = 0.5 * (e_dense + e_dense.permute(1, 0, 2))
            e = e_dense[edge_index_global_lig[0], edge_index_global_lig[1], :]
        else:
            e_dense = torch.zeros(n, n, e.size(-1), device=e.device)
            e_dense[edge_index_global[0], edge_index_global[1], :] = e
            e_dense = 0.5 * (e_dense + e_dense.permute(1, 0, 2))
            e = e_dense[edge_index_global[0], edge_index_global[1], :]

        f = s[i] + s[j] + self.bond_mapping(e)
        edge = torch.cat([f, d], dim=-1)

        bonds_pred = F.silu(self.bonds_lin_0(edge))
        bonds_pred = self.bonds_lin_1(bonds_pred)

            
        return coords_pred, atoms_pred, bonds_pred, property_pred


class HiddenEdgeDistanceMLP(nn.Module):
    def __init__(self, hn_dim: Tuple[int, int]) -> None:
        super(HiddenEdgeDistanceMLP, self).__init__()
        self.sdim, self.vdim = hn_dim
        self.distance_mlp = nn.Sequential(
            DenseLayer(self.sdim, self.sdim // 2, bias=True, activation=nn.SiLU()),
            DenseLayer(self.sdim // 2, self.sdim // 4, bias=True, activation=nn.SiLU()),
            DenseLayer(self.sdim // 4, 1, bias=True, activation=nn.ReLU()),
        )
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.distance_mlp)

    def forward(
        self,
        x: Dict,
        batch_ligand: Tensor,
        batch_pocket: Tensor,
        pocket_mask: Tensor,
        ca_mask: Tensor,
    ) -> Dict:
        s_ligand, s_pocket = (
            x["s"][pocket_mask.squeeze()],
            x["s"][~pocket_mask.squeeze()],
        )
        # select c-alpha representatives
        batch_pocket = batch_pocket[ca_mask]
        s_pocket = s_pocket[ca_mask]
        # create cross indices between ligand and c-alpha
        adj_cross = (batch_ligand[:, None] == batch_pocket[None, :]).nonzero().T
        l, p = adj_cross
        s = s_ligand[l] + s_pocket[p]
        s = self.distance_mlp(s).squeeze(dim=-1)
        return s


class PropertyPredictionHead(nn.Module):
    def __init__(
        self,
        hn_dim: Tuple[int, int],
        num_context_features: int,
    ) -> None:
        super(PropertyPredictionHead, self).__init__()
        self.sdim, self.vdim = hn_dim
        self.num_context_features = num_context_features

        self.scalar_mapping = DenseLayer(
            self.sdim, self.sdim, bias=True, activation=nn.SiLU()
        )
        self.vector_mapping = DenseLayer(
            in_features=self.vdim, out_features=self.sdim, bias=False
        )

        self.output_network = nn.ModuleList(
            [
                GatedEquivariantBlock(
                    self.sdim,
                    self.sdim // 2,
                    activation="silu",
                    scalar_activation=True,
                ),
                GatedEquivariantBlock(
                    self.sdim // 2, num_context_features, activation="silu"
                ),
            ]
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.scalar_mapping.reset_parameters()
        self.vector_mapping.reset_parameters()

    def forward(self, x: Dict, batch: Tensor, edge_index_global: Tensor) -> Dict:
        s, v, p, e = x["s"], x["v"], x["p"], x["e"]

        s = self.scalar_mapping(s)
        v = self.vector_mapping(v) + p.sum() * 0 + e.sum() * 0
        for layer in self.output_network:
            s, v = layer(s, v)
        # include v in output to make sure all parameters have a gradient
        out = s + v.sum() * 0

        return out


class PropertyPredictionMLP(nn.Module):
    def __init__(
        self,
        hn_dim: Tuple[int, int],
        num_context_features: int,
        activation: Union[Callable, nn.Module] = None,
    ) -> None:
        super(PropertyPredictionMLP, self).__init__()
        self.sdim, self.vdim = hn_dim
        self.num_context_features = num_context_features

        self.scalar_mapping = DenseLayer(
            self.sdim,
            self.sdim,
            bias=True,
            activation=nn.SiLU(),
        )
        self.edge_mapping = DenseLayer(
            self.sdim,
            self.sdim,
            bias=True,
            activation=nn.SiLU(),
        )
        self.vector_mapping = DenseLayer(
            in_features=self.vdim, out_features=1, bias=False
        )
        self.property_mapping = DenseLayer(
            self.sdim,
            1,
            bias=True,
            activation=activation,
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.scalar_mapping.reset_parameters()
        self.edge_mapping.reset_parameters()
        self.vector_mapping.reset_parameters()
        self.property_mapping.reset_parameters()

    def forward(self, x: Dict, batch: Tensor, edge_index_global: Tensor) -> Dict:
        s, v, p, e = x["s"], x["v"], x["p"], x["e"]
        e = 0.5 * scatter_mean(
            e,
            index=edge_index_global[1],
            dim=0,
            dim_size=s.size(0),
        )
        s = self.scalar_mapping(s)
        e = self.edge_mapping(e)
        v = self.vector_mapping(v) + p.sum() * 0
        v = torch.norm(v, dim=-1)

        out = self.property_mapping(s + e + v)

        return out


class DenseLayer(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        activation: Union[Callable, nn.Module] = None,
        weight_init: Callable = kaiming_uniform_,
        bias_init: Callable = zeros_,
    ):
        self.weight_init = weight_init
        self.bias_init = bias_init
        super(DenseLayer, self).__init__(in_features, out_features, bias)

        self.activation = activation
        if self.activation is None:
            self.activation = nn.Identity()

    def reset_parameters(self):
        # https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py#L106
        self.weight_init(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            self.bias_init(self.bias)

    def forward(self, input: torch.Tensor):
        y = F.linear(input, self.weight, self.bias)
        y = self.activation(y)
        return y


class GatedEquivBlock(nn.Module):
    def __init__(
        self,
        in_dims: Tuple[int, int],
        out_dims: Tuple[int, Optional[int]],
        hs_dim: Optional[int] = None,
        hv_dim: Optional[int] = None,
        norm_eps: float = 1e-6,
        use_mlp: bool = False,
    ):
        super(GatedEquivBlock, self).__init__()
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.vo = 0 if self.vo is None else self.vo

        self.hs_dim = hs_dim or max(self.si, self.so)
        self.hv_dim = hv_dim or max(self.vi, self.vo)
        self.norm_eps = norm_eps

        self.use_mlp = use_mlp

        self.Wv0 = DenseLayer(self.vi, self.hv_dim + self.vo, bias=False)

        if not use_mlp:
            self.Ws = DenseLayer(self.hv_dim + self.si, self.vo + self.so, bias=True)
        else:
            self.Ws = nn.Sequential(
                DenseLayer(
                    self.hv_dim + self.si, self.si, bias=True, activation=nn.SiLU()
                ),
                DenseLayer(self.si, self.vo + self.so, bias=True),
            )
            if self.vo > 0:
                self.Wv1 = DenseLayer(self.vo, self.vo, bias=False)
            else:
                self.Wv1 = None

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.Ws)
        reset(self.Wv0)
        if self.use_mlp:
            if self.vo > 0:
                reset(self.Wv1)

    def forward(self, x: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        s, v = x
        vv = self.Wv0(v)

        if self.vo > 0:
            vdot, v = vv.split([self.hv_dim, self.vo], dim=-1)
        else:
            vdot = vv

        vdot = torch.clamp(torch.pow(vdot, 2).sum(dim=1), min=self.norm_eps)  # .sqrt()

        s = torch.cat([s, vdot], dim=-1)
        s = self.Ws(s)
        if self.vo > 0:
            gate, s = s.split([self.vo, self.so], dim=-1)
            v = gate.unsqueeze(1) * v
            if self.use_mlp:
                v = self.Wv1(v)

        return s, v


class LayerNorm(nn.Module):
    def __init__(
        self,
        dims: Tuple[int, Optional[int]],
        eps: float = 1e-6,
        affine: bool = True,
        latent_dim=None,
    ):
        super().__init__()

        self.dims = dims
        self.sdim, self.vdim = dims
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.Tensor(self.sdim))
            self.bias = nn.Parameter(torch.Tensor(self.sdim))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            self.weight.data.fill_(1.0)
            self.bias.data.fill_(0.0)

    def forward(self, x: Dict, batch: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        s, v = x.get("s"), x.get("v")
        batch_size = int(batch.max()) + 1
        smean = s.mean(dim=-1, keepdim=True)
        smean = scatter_mean(smean, batch, dim=0, dim_size=batch_size)

        s = s - smean[batch]

        var = (s * s).mean(dim=-1, keepdim=True)
        var = scatter_mean(var, batch, dim=0, dim_size=batch_size)
        var = torch.clamp(var, min=self.eps)  # .sqrt()
        sout = s / var[batch]

        if self.weight is not None and self.bias is not None:
            sout = sout * self.weight + self.bias

        if v is not None:
            vmean = torch.pow(v, 2).sum(dim=1, keepdim=True).mean(dim=-1, keepdim=True)
            vmean = scatter_mean(vmean, batch, dim=0, dim_size=batch_size)
            vmean = torch.clamp(vmean, min=self.eps)
            vout = v / vmean[batch]
        else:
            vout = None

        out = sout, vout

        return out

    def __repr__(self):
        return f"{self.__class__.__name__}(dims={self.dims}, " f"affine={self.affine})"


class AdaptiveLayerNorm(nn.Module):
    def __init__(
        self,
        dims: Tuple[int, Optional[int]],
        latent_dim: int,
        eps: float = 1e-6,
        affine: bool = True,
    ):
        super().__init__()

        self.dims = dims
        self.sdim, self.vdim = dims
        self.latent_dim = latent_dim
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight_bias = DenseLayer(latent_dim, 2 * self.sdim, bias=True)
        else:
            print(
                "Affine was set to False. This layer should used the affine transformation"
            )
            raise ValueError
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        self.weight_bias.bias.data[: self.sdim] = 1
        self.weight_bias.bias.data[self.sdim :] = 0

    def forward(self, x: Dict, batch: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        s, v, z = x["s"], x["v"], x["z"]
        batch_size = int(batch.max()) + 1

        smean = s.mean(dim=-1, keepdim=True)
        smean = scatter_mean(smean, batch, dim=0, dim_size=batch_size)
        s = s - smean[batch]
        var = (s * s).mean(dim=-1, keepdim=True)
        var = scatter_mean(var, batch, dim=0, dim_size=batch_size)
        var = torch.clamp(var, min=self.eps)  # .sqrt()
        sout = s / var[batch]

        weight, bias = self.weight_bias(z).chunk(2, dim=-1)
        sout = sout * weight[batch] + bias[batch]

        if v is not None:
            vmean = torch.pow(v, 2).sum(dim=1, keepdim=True).mean(dim=-1, keepdim=True)
            vmean = scatter_mean(vmean, batch, dim=0, dim_size=batch_size)
            vmean = torch.clamp(vmean, min=self.eps)
            vout = v / vmean[batch]
        else:
            vout = None

        out = sout, vout

        return out

    def __repr__(self):
        return f"{self.__class__.__name__}(dims={self.dims}, " f"affine={self.affine})"


class SE3Norm(nn.Module):
    def __init__(self, eps: float = 1e-5, device=None, dtype=None) -> None:
        """Note: There is a relatively similar layer implemented by NVIDIA:
        https://catalog.ngc.nvidia.com/orgs/nvidia/resources/se3transformer_for_pytorch.
        It computes a ReLU on a mean-zero normalized norm, which I find surprising.
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.normalized_shape = (1, 1)  # type: ignore[arg-type]
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(self.normalized_shape, **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.ones_(self.weight)

    def forward(
        self,
        pos: Tensor,
        batch: Tensor,
        batch_lig: Tensor = None,
        pocket_mask: Tensor = None,
    ):
        if pocket_mask is not None:
            norm = torch.norm(pos, dim=-1, keepdim=True) * pocket_mask  # n, 1
        else:
            norm = torch.norm(pos, dim=-1, keepdim=True)
        batch_size = int(batch.max()) + 1
        if batch_lig is not None:
            n_nodes_lig = batch_lig.bincount()
            mean_norm = scatter_add(norm, batch, dim=0, dim_size=batch_size)
            mean_norm = mean_norm / n_nodes_lig.unsqueeze(1)
        else:
            mean_norm = scatter_mean(norm, batch, dim=0, dim_size=batch_size)
        new_pos = self.weight * pos / (mean_norm[batch] + self.eps)
        return new_pos

    def extra_repr(self) -> str:
        return "{normalized_shape}, eps={eps}".format(**self.__dict__)


act_class_mapping = {
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
}


class GatedEquivariantBlock(nn.Module):
    """Gated Equivariant Block as defined in Schütt et al. (2021):
    Equivariant message passing for the prediction of tensorial properties and molecular spectra
    """

    def __init__(
        self,
        hidden_channels,
        out_channels,
        intermediate_channels=None,
        activation="silu",
        scalar_activation=False,
    ):
        super(GatedEquivariantBlock, self).__init__()
        self.out_channels = out_channels

        if intermediate_channels is None:
            intermediate_channels = hidden_channels

        self.vec1_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.vec2_proj = nn.Linear(hidden_channels, out_channels, bias=False)

        act_class = act_class_mapping[activation]
        self.update_net = nn.Sequential(
            nn.Linear(hidden_channels * 2, intermediate_channels),
            act_class(),
            nn.Linear(intermediate_channels, out_channels * 2),
        )

        self.act = act_class() if scalar_activation else None

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.vec1_proj.weight)
        nn.init.xavier_uniform_(self.vec2_proj.weight)
        nn.init.xavier_uniform_(self.update_net[0].weight)
        self.update_net[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.update_net[2].weight)
        self.update_net[2].bias.data.fill_(0)

    def forward(self, x, v):
        vec1_buffer = self.vec1_proj(v)

        # detach zero-entries to avoid NaN gradients during force loss backpropagation
        vec1 = torch.zeros(
            vec1_buffer.size(0), vec1_buffer.size(2), device=vec1_buffer.device
        )
        mask = (vec1_buffer != 0).view(vec1_buffer.size(0), -1).any(dim=1)
        vec1[mask] = torch.norm(vec1_buffer[mask], dim=-2)

        vec2 = self.vec2_proj(v)

        x = torch.cat([x, vec1], dim=-1)
        x, v = torch.split(self.update_net(x), self.out_channels, dim=-1)
        v = v.unsqueeze(1) * vec2

        if self.act is not None:
            x = self.act(x)
        return x, v
