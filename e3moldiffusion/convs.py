from typing import Optional, Tuple

import torch
from torch import Tensor, nn
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import reset
from torch_geometric.typing import OptTensor
from torch_scatter import scatter
from torch_scatter.composite import scatter_softmax

from e3moldiffusion.modules import DenseLayer, GatedEquivBlock, ChebyshevExpansion, PolynomialCutoff, BesselExpansion


def cross_product(a: Tensor, b: Tensor, dim: int) -> Tensor:
    if a.dtype != torch.float16 and b.dtype != torch.float16:
        return torch.linalg.cross(a, b, dim=dim)
    else:
        s1 = a[:, 1, :] * b[:, -1, :] - a[:, -1, :] * b[:, 1, :]
        s2 = a[:, -1, :] * b[:, 0, :] - a[:, 0, :] * b[:, -1, :]
        s3 = a[:, 0, :] * b[:, 1, :] - a[:, 1, :] * b[:, 0, :]
        cross = torch.stack([s1, s2, s3], dim=dim)
        return cross


class EQGATConv(MessagePassing):
    """
    Slightly modified SO(3) equivariant graph attention convolution described in
    @inproceedings{
        le2022representation,
        title={Representation Learning on Biomolecular Structures using Equivariant Graph Attention},
        author={Tuan Le and Frank Noe and Djork-Arn{\'e} Clevert},
        booktitle={The First Learning on Graphs Conference},
        year={2022},
        url={https://openreview.net/forum?id=kv4xUo5Pu6}
    }
    """
    def __init__(
        self,
        in_dims: Tuple[int, Optional[int]],
        out_dims: Tuple[int, Optional[int]],
        edge_dim: Optional[int] = None,
        eps: float = 1e-6,
        has_v_in: bool = False,
        use_mlp_update: bool = True,
        vector_aggr: str = "mean",
        use_cross_product: bool = True
    ):
        super(EQGATConv, self).__init__(
            node_dim=0, aggr=None, flow="source_to_target"
        )
        self.vector_aggr = vector_aggr
        self.in_dims = in_dims
        self.si, self.vi = in_dims
        self.out_dims = out_dims
        self.so, self.vo = out_dims
        self.has_v_in = has_v_in
        self.use_cross_product = use_cross_product
        if has_v_in:
            self.vector_net = DenseLayer(self.vi, self.vi, bias=False)
            self.v_mul = 3 if use_cross_product else 2
        else:
            self.v_mul = 1
            self.vector_net = nn.Identity()

        if edge_dim is None:
            edge_dim = 0

        self.edge_net = nn.Sequential(DenseLayer(2 * self.si + edge_dim + 1,
                                                 self.si,
                                                 bias=True, activation=nn.SiLU()
                                                 ),
                                      DenseLayer(self.si, self.v_mul * self.vi + self.si,
                                                 bias=True
                                                 )
                                      )
        self.scalar_net = DenseLayer(self.si, self.si, bias=True)
        self.update_net = GatedEquivBlock(in_dims=(self.si, self.vi),
                                          hs_dim=self.si, hv_dim=self.vi,
                                          out_dims=(self.so, self.vo),
                                          norm_eps=eps,
                                          use_mlp=use_mlp_update
                                          )
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.edge_net)
        if self.has_v_in:
            reset(self.vector_net)
        reset(self.scalar_net)
        reset(self.update_net)

    def forward(
        self,
        x: Tuple[Tensor, Tensor],
        edge_index: Tensor,
        edge_attr: Tuple[Tensor, Tensor, OptTensor],
    ):

        s, v = x
        d, r, e = edge_attr

        ms, mv = self.propagate(
            sa=s,
            sb=self.scalar_net(s),
            va=v,
            vb=self.vector_net(v),
            edge_attr=(d, r, e),
            edge_index=edge_index,
            dim_size=s.size(0),
        )

        s = ms + s
        v = mv + v

        ms, mv = self.update_net(x=(s, v))

        s = ms + s
        v = mv + v

        return s, v

    def aggregate(
        self,
            inputs: Tuple[Tensor, Tensor],
            index: Tensor,
            dim_size: Optional[int] = None
    ) -> Tuple[Tensor, Tensor]:
        s = scatter(inputs[0], index=index, dim=0, reduce="add", dim_size=dim_size)
        v = scatter(inputs[1], index=index, dim=0, reduce=self.vector_aggr, dim_size=dim_size)
        return s, v

    def message(
        self,
        sa_i: Tensor,
        sa_j: Tensor,
        sb_j: Tensor,
        va_i: Tensor,
        va_j: Tensor,
        vb_j: Tensor,
        index: Tensor,
        edge_attr: Tuple[Tensor, Tensor, OptTensor],
        dim_size: Optional[int]
    ) -> Tuple[Tensor, Tensor]:

        d, r, e = edge_attr

        # de = d.view(-1, 1)
        de = 1.0 / (1.0 + d.view(-1, 1))

        if e is not None:
            aij = torch.cat([sa_i, sa_j, de, e], dim=-1)
        else:
            aij = torch.cat([sa_i, sa_j, de], dim=-1)

        aij = self.edge_net(aij)

        if self.has_v_in:
            aij, vij0 = aij.split([self.si, self.v_mul*self.vi], dim=-1)
            vij0 = vij0.unsqueeze(1)
            if self.use_cross_product:
                vij0, vij1, vij2 = vij0.chunk(3, dim=-1)
            else:
                vij0, vij1 = vij0.chunk(2, dim=-1)
        else:
            aij, vij0 = aij.split([self.si, self.vi], dim=-1)
            vij0 = vij0.unsqueeze(1)

        # feature attention
        aij = scatter_softmax(aij, index=index, dim=0, dim_size=dim_size)
        ns_j = aij * sb_j
        nv0_j = r.unsqueeze(-1) * vij0

        if self.has_v_in:
            nv1_j = vij1 * vb_j
            if self.use_cross_product:
                v_ij_cross = cross_product(va_i, va_j, dim=1)
                nv2_j = vij2 * v_ij_cross
                nv_j = nv0_j + nv1_j + nv2_j
            else:
                nv_j = nv0_j + nv1_j
        else:
            nv_j = nv0_j

        return ns_j, nv_j
    
class EQGATRBFConv(MessagePassing):
    """
    Slightly modified SO(3) equivariant graph attention convolution described in
    @inproceedings{
        le2022representation,
        title={Representation Learning on Biomolecular Structures using Equivariant Graph Attention},
        author={Tuan Le and Frank Noe and Djork-Arn{\'e} Clevert},
        booktitle={The First Learning on Graphs Conference},
        year={2022},
        url={https://openreview.net/forum?id=kv4xUo5Pu6}
    }
    """
    def __init__(
        self,
        in_dims: Tuple[int, Optional[int]],
        out_dims: Tuple[int, Optional[int]],
        rbf_dim: int,
        cutoff: float = 10.0,
        eps: float = 1e-6,
        has_v_in: bool = False,
        use_mlp_update: bool = True,
        vector_aggr: str = "mean",
        use_cross_product: bool = True
    ):
        super(EQGATRBFConv, self).__init__(
            node_dim=0, aggr=None, flow="source_to_target"
        )
        self.vector_aggr = vector_aggr
        self.in_dims = in_dims
        self.si, self.vi = in_dims
        self.out_dims = out_dims
        self.so, self.vo = out_dims
        self.has_v_in = has_v_in
        self.use_cross_product = use_cross_product
        if has_v_in:
            self.vector_net = DenseLayer(self.vi, self.vi, bias=False)
            self.v_mul = 3 if use_cross_product else 2
        else:
            self.v_mul = 1
            self.vector_net = nn.Identity()

        self.rbf_dim = rbf_dim
        self.cutoff = cutoff
    
        self.cutoff_fnc = PolynomialCutoff(cutoff=cutoff, p=6)
        self.rbf = BesselExpansion(cutoff, rbf_dim)
  
        self.edge_net = nn.Sequential(DenseLayer(2 * self.si + rbf_dim + 1,
                                                 self.si,
                                                 bias=True, activation=nn.SiLU()
                                                 ),
                                      DenseLayer(self.si, self.v_mul * self.vi + self.si,
                                                 bias=True
                                                 )
                                      )
        self.scalar_net = DenseLayer(self.si, self.si, bias=True)
        self.update_net = GatedEquivBlock(in_dims=(self.si, self.vi),
                                          hs_dim=self.si, hv_dim=self.vi,
                                          out_dims=(self.so, self.vo),
                                          norm_eps=eps,
                                          use_mlp=use_mlp_update
                                          )
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.edge_net)
        if self.has_v_in:
            reset(self.vector_net)
        reset(self.scalar_net)
        reset(self.update_net)

    def forward(
        self,
        x: Tuple[Tensor, Tensor],
        edge_index: Tensor,
        edge_attr: Tuple[Tensor, Tensor, OptTensor],
    ):

        s, v = x
        d, r, e = edge_attr

        ms, mv = self.propagate(
            sa=s,
            sb=self.scalar_net(s),
            va=v,
            vb=self.vector_net(v),
            edge_attr=(d, r, e),
            edge_index=edge_index,
            dim_size=s.size(0),
        )

        s = ms + s
        v = mv + v

        ms, mv = self.update_net(x=(s, v))

        s = ms + s
        v = mv + v

        return s, v

    def aggregate(
        self,
            inputs: Tuple[Tensor, Tensor],
            index: Tensor,
            dim_size: Optional[int] = None
    ) -> Tuple[Tensor, Tensor]:
        s = scatter(inputs[0], index=index, dim=0, reduce="add", dim_size=dim_size)
        v = scatter(inputs[1], index=index, dim=0, reduce=self.vector_aggr, dim_size=dim_size)
        return s, v

    def message(
        self,
        sa_i: Tensor,
        sa_j: Tensor,
        sb_j: Tensor,
        va_i: Tensor,
        va_j: Tensor,
        vb_j: Tensor,
        index: Tensor,
        edge_attr: Tuple[Tensor, Tensor, OptTensor],
        dim_size: Optional[int]
    ) -> Tuple[Tensor, Tensor]:

        d, r, _ = edge_attr

        # de = d.view(-1, 1)
        de = 1.0 / (1.0 + d.view(-1, 1))        
        rbf = self.rbf(d)
        cutoff = self.cutoff_fnc(d).view(-1, 1)
        rbf = cutoff * rbf

        aij = torch.cat([sa_i, sa_j, de, rbf], dim=-1)

        aij = self.edge_net(aij)

        if self.has_v_in:
            aij, vij0 = aij.split([self.si, self.v_mul*self.vi], dim=-1)
            vij0 = vij0.unsqueeze(1)
            if self.use_cross_product:
                vij0, vij1, vij2 = vij0.chunk(3, dim=-1)
            else:
                vij0, vij1 = vij0.chunk(2, dim=-1)
        else:
            aij, vij0 = aij.split([self.si, self.vi], dim=-1)
            vij0 = vij0.unsqueeze(1)

        # feature attention
        aij = scatter_softmax(aij, index=index, dim=0, dim_size=dim_size)
        ns_j = aij * sb_j
        nv0_j = r.unsqueeze(-1) * vij0

        if self.has_v_in:
            nv1_j = vij1 * vb_j
            if self.use_cross_product:
                v_ij_cross = cross_product(va_i, va_j, dim=1)
                nv2_j = vij2 * v_ij_cross
                nv_j = nv0_j + nv1_j + nv2_j
            else:
                nv_j = nv0_j + nv1_j
        else:
            nv_j = nv0_j

        return ns_j, nv_j
    