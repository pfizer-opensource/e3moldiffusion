from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import reset
from torch_geometric.typing import OptTensor
from torch_scatter import scatter
from torch_scatter.composite import scatter_softmax

from e3moldiffusion.modules import DenseLayer, GatedEquivBlock, LayerNorm


def cross_product(a: Tensor, b: Tensor, dim: int) -> Tensor:
    if a.dtype != torch.float16 and b.dtype != torch.float16:
        return torch.linalg.cross(a, b, dim=dim)
    else:
        s1 = a[:, 1, :] * b[:, -1, :] - a[:, -1, :] * b[:, 1, :]
        s2 = a[:, -1, :] * b[:, 0, :] - a[:, 0, :] * b[:, -1, :]
        s3 = a[:, 0, :] * b[:, 1, :] - a[:, 1, :] * b[:, 0, :]
        cross = torch.stack([s1, s2, s3], dim=dim)
        return cross


class EQGATFCConv(MessagePassing):
    def __init__(
        self,
        in_dims: Tuple[int, Optional[int]],
        out_dims: Tuple[int, Optional[int]],
        eps: float = 1e-6,
        has_v_in: bool = False,
        use_mlp_update: bool = True,
        vector_aggr: str = "mean",
        use_cross_product: bool = False
    ):
        super(EQGATFCConv, self).__init__(
            node_dim=0, aggr=None, flow="source_to_target"
        )
        
        self.vector_aggr = vector_aggr
        self.in_dims = in_dims
        self.si, self.vi = in_dims
        self.out_dims = out_dims
        self.so, self.vo = out_dims
        self.has_v_in = has_v_in
        self.use_cross_product = use_cross_product
        self.silu = nn.SiLU()
        if has_v_in:
            self.vector_net = DenseLayer(self.vi, self.vi, bias=False)
            self.v_mul = 3 if use_cross_product else 2
        else:
            self.v_mul = 1
            self.vector_net = nn.Identity()

        self.edge_net = nn.Sequential(DenseLayer(self.si + 2, self.si, bias=True, activation=nn.SiLU()),
                                      DenseLayer(self.si, self.v_mul * self.vi + self.si + 1, bias=True)
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
        edge_attr: Tuple[Tensor, Tensor, Tensor, OptTensor],
    ):
        
        s, v, p = x
        d, a, r, e = edge_attr
    
        ms, mv, mp = self.propagate(
            sa=s,
            sb=self.scalar_net(s),
            va=v,
            vb=self.vector_net(v),
            edge_attr=(d, a, r, e),
            edge_index=edge_index,
            dim_size=s.size(0),
        )

        s = ms + s
        v = mv + v
        p = mp + p

        ms, mv = self.update_net(x=(s, v))

        s = ms + s
        v = mv + v
    
        out = {"s": s, "v": v, 'p': p}
        return out

    def aggregate(
        self,
            inputs: Tuple[Tensor, Tensor, Tensor],
            index: Tensor,
            dim_size: Optional[int] = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        s = scatter(inputs[0], index=index, dim=0, reduce="add", dim_size=dim_size)
        v = scatter(inputs[1], index=index, dim=0, reduce=self.vector_aggr, dim_size=dim_size)
        p = scatter(inputs[2], index=index, dim=0, reduce=self.vector_aggr, dim_size=dim_size)
        return s, v, p

    def message(
        self,
        sa_i: Tensor,
        sa_j: Tensor,
        sb_j: Tensor,
        va_i: Tensor,
        va_j: Tensor,
        vb_j: Tensor,
        index: Tensor,
        edge_attr: Tuple[Tensor, Tensor, Tensor, OptTensor],
        dim_size: Optional[int]
    ) -> Tuple[Tensor, Tensor]:

        d, a, r, _ = edge_attr

        de0 = d.view(-1, 1)
        a0 = a.view(-1, 1)
    
        aij = torch.cat([sa_i + sa_j, de0, a0], dim=-1)
        aij = self.edge_net(aij)
        
        fdim = aij.shape[-1]
        aij, gij = aij.split([fdim - 1, 1], dim=-1)
        fdim = aij.shape[-1]
        pj = gij * r

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

        return ns_j, nv_j, pj
    

class EQGATGNN(nn.Module):
    def __init__(self,
                 hn_dim: Tuple[int, int] = (64, 16),
                 num_layers: int = 5,
                 use_norm: bool = True,
                 use_cross_product: bool = False,
                 vector_aggr: str = "mean"
                 ):
        super(EQGATGNN, self).__init__()

     
        self.num_layers = num_layers
        self.sdim, self.vdim = hn_dim
        
        convs = []
        
        for i in range(num_layers):
            # ONLY FULLY-CONNECTED 
            convs.append(
                EQGATFCConv(in_dims=hn_dim,
                            out_dims=hn_dim,
                            has_v_in=i>0,
                            use_mlp_update= i < (num_layers - 1),
                            vector_aggr=vector_aggr,
                            use_cross_product=use_cross_product
                            )
                )
        
        self.convs = nn.ModuleList(convs)
        self.use_norm = use_norm
        self.norms = nn.ModuleList([
            LayerNorm(dims=hn_dim) if use_norm else nn.Identity()
            for _ in range(num_layers)
        ])
        
        self.reset_parameters()

    def reset_parameters(self):
        for conv, norm in zip(self.convs, self.norms):
            conv.reset_parameters()
            if self.use_norm:
                norm.reset_parameters()
    
    def calculate_edge_attrs(self, edge_index: Tensor, edge_attr: OptTensor, pos: Tensor):
        source, target = edge_index
        r = pos[target] - pos[source]
        a = pos[target] * pos[source]
        a = a.sum(-1)
        d = torch.clamp(torch.pow(r, 2).sum(-1), min=1e-6).sqrt()
        r_norm = torch.div(r, d.unsqueeze(-1))
        edge_attr = (d, a, r_norm, edge_attr)
        return edge_attr
    
    def forward(self,
                s: Tensor,
                v: Tensor,
                p: Tensor,
                edge_index: Tensor,
                edge_attr: Tuple[Tensor, Tensor, Tensor, OptTensor],
                batch: Tensor = None) -> Dict:
    
        # edge_attr_xyz (distances, cosines, relative_positions, edge_features)
        # (E, E, E x 3, E x F)
        
        for i in range(len(self.convs)):   
            out = self.convs[i](x=(s, v, p), edge_index=edge_index, edge_attr=edge_attr)
            s, v, p = out["s"], out["v"], out['p']
            if self.use_norm:
                s, v = self.norms[i](x=(s, v), batch=batch)
            
        out = {"s": s, "v": v, 'p': p}
        
        return out



class PredictionHead(nn.Module):
    def __init__(self, hn_dim: Tuple[int, int], num_atom_types: int) -> None:
        super(PredictionHead, self).__init__()
        self.sdim, self.vdim = hn_dim
        self.num_atom_types = num_atom_types
        
        self.shared_mapping = DenseLayer(self.sdim, self.sdim, bias=True, activation=nn.SiLU())
        self.coords_lin = DenseLayer(in_features=self.vdim, out_features=2, bias=False)
        self.atoms_lin = DenseLayer(in_features=self.sdim, out_features=num_atom_types, bias=True)
        self.reset_parameters()
        
    def reset_parameters(self):
        self.shared_mapping.reset_parameters()
        self.coords_lin.reset_parameters()
        self.atoms_lin.reset_parameters()
        
    def forward(self,
                x: Dict,
                ) -> Dict:
        
        s, v, p = x["s"], x["v"], x["p"]
        s = self.shared_mapping(s)
        
        dm, pos_std = self.coords_lin(v).chunk(2, dim=-1)
        pos_std = pos_std.squeeze()
        atoms_pred = self.atoms_lin(s)
        
        pos_mean = p.squeeze() + dm.squeeze()
        
        out = {"coords_mean": pos_mean, "coords_std": pos_std,
               "atoms_pred": atoms_pred
               }
        
        return out

class E3ARDiffusionModel(nn.Module):
    def __init__(self,
                 num_atom_types: int,
                 hn_dim: Tuple[int, int] = (64, 16),
                 num_layers: int = 5,
                 use_norm: bool = True,
                 use_cross_product: bool = True,
                 vector_aggr: str = "mean"
                 ) -> None:
        super(E3ARDiffusionModel, self).__init__()
        
        self.time_mapping = DenseLayer(1, hn_dim[0], bias=True)
        self.atom_mapping = DenseLayer(num_atom_types + 1, hn_dim[0], bias=True)
        self.atom_time_mapping = DenseLayer(hn_dim[0], hn_dim[0], bias=True)
        
        self.sdim, self.vdim = hn_dim

        self.gnn = EQGATGNN(
            hn_dim=hn_dim,
            num_layers=num_layers,
            use_norm=use_norm,
            use_cross_product=use_cross_product,
            vector_aggr=vector_aggr,
        )
        
        self.prediction_head = PredictionHead(hn_dim=hn_dim, num_atom_types=num_atom_types)
       
        self.reset_parameters()

    def reset_parameters(self):
        self.atom_mapping.reset_parameters()
        self.time_mapping.reset_parameters()
        self.atom_time_mapping.reset_parameters()
        self.gnn.reset_parameters()
        self.prediction_head.reset_parameters()
        
    def calculate_edge_attrs(self, edge_index: Tensor, edge_attr: OptTensor, pos: Tensor):
        source, target = edge_index
        r = pos[target] - pos[source]
        a = pos[target] * pos[source]
        a = a.sum(-1)
        d = torch.clamp(torch.pow(r, 2).sum(-1), min=1e-6).sqrt()
        r_norm = torch.div(r, d.unsqueeze(-1))
        edge_attr = (d, a, r_norm, edge_attr)
        return edge_attr
    
    def forward(
        self,
        x: Tensor,
        t: Tensor,
        mask: Tensor,
        pos: Tensor,
        edge_index: Tensor,
        edge_attr: OptTensor = None,
        batch: OptTensor = None) -> Dict[Tensor, Tensor]:
        
        if batch is None:
            batch = torch.zeros(x.size(0), device=x.device, dtype=torch.long)
                    
        x_m = x * mask.view(-1, 1)
        pos_m = pos * mask.view(-1, 1)
        
        edge_attr = self.calculate_edge_attrs(edge_index=edge_index, edge_attr=edge_attr, pos=pos)
        
        t = self.time_mapping(t)
        mask_f = (1.0 - mask.float()).unsqueeze(-1)
        feat = torch.cat([x_m, mask_f], dim=-1)
        s = self.atom_mapping(feat)
        s = self.atom_time_mapping(s + t)
        
        v = torch.zeros(size=(x.size(0), 3, self.vdim), device=s.device)

        out = self.gnn(
            s=s, v=v, p=pos_m,
            edge_index=edge_index, edge_attr=edge_attr,
            batch=batch
        )
        
        pred = self.prediction_head(x=out)
             
        return pred
    
    
if __name__ == "__main__":
    pass

