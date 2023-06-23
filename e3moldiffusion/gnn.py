from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn
from torch_geometric.nn import radius_graph
from torch_geometric.typing import OptTensor
from torch_scatter import scatter_mean, scatter_add

from e3moldiffusion.convs import (EQGATGlobalConvFinal,
                                  EQGATGlobalEdgeConvFinal,
                                  EQGATLocalConvFinal, EQGATLocalEdgeConvFinal)
from e3moldiffusion.modules import LayerNorm, DenseLayer


class EQGATGNN(nn.Module):
    """_summary_
    EQGAT GNN Network updating node-level scalar, vectors and position features.
    No edge-features are modelled in this class.
    Args:
        nn (_type_): _description_
    """
    def __init__(self,
                 hn_dim: Tuple[int, int] = (64, 16),
                 rbf_dim: int = 64,
                 cutoff_local: float = 5.0,
                 num_layers: int = 5,
                 use_norm: bool = True,
                 use_cross_product: bool = False,
                 vector_aggr: str = "mean",
                 fully_connected: bool = False,
                 local_global_model: bool = True,
                 recompute_radius_graph: bool = False,
                 recompute_edge_attributes: bool = False
                 ):
        super(EQGATGNN, self).__init__()

        self.num_layers = num_layers
        self.fully_connected = fully_connected
        self.local_global_model = local_global_model
        self.recompute_radius_graph = recompute_radius_graph
        self.recompute_edge_attributes = recompute_edge_attributes
        
        self.sdim, self.vdim = hn_dim
        
        convs = []
        
        for i in range(num_layers):
            if fully_connected:
                convs.append(
                    EQGATGlobalConvFinal(in_dims=hn_dim,
                                         out_dims=hn_dim,
                                         has_v_in=i>0,
                                         use_mlp_update= i < (num_layers - 1),
                                         vector_aggr=vector_aggr,
                                         use_cross_product=use_cross_product
                                         )
                )
            else:
                if (i == self.num_layers - 2 or i == 0) and local_global_model:
                    convs.append(
                        EQGATGlobalConvFinal(in_dims=hn_dim,
                                             out_dims=hn_dim,
                                             has_v_in=i>0,
                                             use_mlp_update= i < (num_layers - 1),
                                             vector_aggr=vector_aggr,
                                             use_cross_product=use_cross_product
                                             )
                        )
                else:
                    convs.append(
                        EQGATLocalConvFinal(in_dims=hn_dim,
                                            out_dims=hn_dim,
                                            rbf_dim=rbf_dim,
                                            cutoff=cutoff_local,
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
                edge_index_local: Tensor,
                edge_attr_local: Tuple[Tensor, Tensor, Tensor, OptTensor],
                edge_index_global: Tensor,
                edge_attr_global: Tuple[Tensor, Tensor, Tensor, OptTensor],
                batch: Tensor = None) -> Dict:
        # edge_attr_xyz (distances, cosines, relative_positions, edge_features)
        # (E, E, E x 3, None)
        
        for i in range(len(self.convs)):   
            
            if self.fully_connected:
                edge_index_in = edge_index_global
                edge_attr_in = edge_attr_global
            else:
                if (i == self.num_layers - 2 or i == 0) and self.local_global_model:
                    edge_index_in = edge_index_global
                    edge_attr_in = edge_attr_global
                else:
                    edge_index_in = edge_index_local
                    edge_attr_in = edge_attr_local
                        
            if self.use_norm:
                s, v = self.norms[i](x=(s, v), batch=batch)
            
            out = self.convs[i](x=(s, v, p), edge_index=edge_index_in, edge_attr=edge_attr_in)
            
            if self.fully_connected:
                s, v, p = out["s"], out["v"], out['p']
                p = p - scatter_mean(p, batch, dim=0)[batch]
                if self.recompute_edge_attributes:
                    edge_attr_global = self.calculate_edge_attrs(edge_index=edge_index_global, pos=p, edge_attr=None)
            else:
                if (i == self.num_layers - 2 or i == 0) and self.local_global_model:
                    s, v, p = out["s"], out["v"], out['p']
                    p = p - scatter_mean(p, batch, dim=0)[batch]
                    if self.recompute_edge_attributes:
                        edge_attr_global = self.calculate_edge_attrs(edge_index=edge_index_global, pos=p, edge_attr=None)
                    if self.recompute_radius_graph:
                        edge_index_local = radius_graph(p, r=self.cutoff_local, batch=batch, max_num_neighbors=128, flow='source_to_target')
                    if self.recompute_edge_attributes:
                        edge_attr_local = self.calculate_edge_attrs(edge_index=edge_index_local, pos=p, edge_attr=None)
                else:
                    s, v, p = out["s"], out["v"], out["p"], out["e"]
                    p = p - scatter_mean(p, batch, dim=0)[batch]
                    if self.recompute_radius_graph:
                        edge_index_local = radius_graph(p, r=self.cutoff_local, batch=batch, max_num_neighbors=128, flow='source_to_target')
                    if self.recompute_edge_attributes:
                        edge_attr_local = self.calculate_edge_attrs(edge_index=edge_index_local, pos=p, edge_attr=None)
                        edge_attr_global = self.calculate_edge_attrs(edge_index=edge_index_global, pos=p, edge_attr=None)


        out = {"s": s, "v": v, "e": None, 'p': p}
        return out


class EQGATEdgeGNN(nn.Module):
    """_summary_
    EQGAT GNN Network updating node-level scalar, vectors and position features as well as edge-features.
    Args:
        nn (_type_): _description_
    """
    def __init__(self,
                 hn_dim: Tuple[int, int] = (64, 16),
                 rbf_dim: int = 64,
                 edge_dim: Optional[int] = 16,
                 cutoff_local: float = 5.0,
                 num_layers: int = 5,
                 use_norm: bool = True,
                 use_cross_product: bool = False,
                 vector_aggr: str = "mean",
                 fully_connected: bool = True,
                 local_global_model: bool = False,
                 recompute_radius_graph: bool = True,
                 recompute_edge_attributes: bool = True
                 ):
        super(EQGATEdgeGNN, self).__init__()

        assert fully_connected or local_global_model
        self.num_layers = num_layers
        self.fully_connected = fully_connected
        self.local_global_model = local_global_model
        self.cutoff_local = cutoff_local
        self.recompute_radius_graph = recompute_radius_graph
        self.recompute_edge_attributes = recompute_edge_attributes
        
        self.sdim, self.vdim = hn_dim
        self.edge_dim = edge_dim
        
        convs = []
        
        for i in range(num_layers):
            if fully_connected:
                convs.append(
                    EQGATGlobalEdgeConvFinal(in_dims=hn_dim,
                                             out_dims=hn_dim,
                                             edge_dim=edge_dim,
                                             has_v_in=i>0,
                                             use_mlp_update= i < (num_layers - 1),
                                             vector_aggr=vector_aggr,
                                             use_cross_product=use_cross_product
                                             )
                )
            else:
                if (i == self.num_layers - 2 or i == 0) and local_global_model:
                    convs.append(
                        EQGATGlobalEdgeConvFinal(in_dims=hn_dim,
                                                 out_dims=hn_dim,
                                                 edge_dim=edge_dim,
                                                 has_v_in=i>0,
                                                 use_mlp_update= i < (num_layers - 1),
                                                 vector_aggr=vector_aggr,
                                                 use_cross_product=use_cross_product
                                                 )
                        )
                else:
                    convs.append(
                        EQGATLocalEdgeConvFinal(in_dims=hn_dim,
                                                out_dims=hn_dim,
                                                rbf_dim=rbf_dim,
                                                cutoff=cutoff_local,
                                                edge_dim=edge_dim,
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
    
    def to_dense_edge_tensor(self, edge_index, edge_attr, num_nodes):
        E = torch.zeros(num_nodes, num_nodes, edge_attr.size(-1), device=edge_attr.device, dtype=edge_attr.dtype)
        E[edge_index[0], edge_index[1], :] = edge_attr
        return E
    
    def from_dense_edge_tensor(self, edge_index, E):
        return E[edge_index[0], edge_index[1], :]
    
    def forward(self,
                s: Tensor,
                v: Tensor,
                p: Tensor,
                edge_index_local: Tensor,
                edge_attr_local: Tuple[Tensor, Tensor, Tensor, Tensor],
                edge_index_global: Tensor,
                edge_attr_global: Tuple[Tensor, Tensor, Tensor, Tensor],
                batch: Tensor = None) -> Dict:
        
        # edge_attr_xyz (distances, cosines, relative_positions, edge_features)
        # (E, E, E x 3, E x F)
        n = s.size(0)
        edge_local_dense = torch.zeros(n, n, self.edge_dim, device=s.device, dtype=torch.float32)
        edge_global_dense = torch.zeros_like(edge_local_dense)
        edge_local_mask = torch.zeros(n, n, 1, device=s.device, dtype=torch.float32)
        edge_local_mask[edge_index_local[0], edge_index_local[1]] = 1.0
        
        
        for i in range(len(self.convs)):   
            if self.fully_connected:
                edge_index_in = edge_index_global
                edge_attr_in = edge_attr_global
            else:
                if (i == self.num_layers - 2 or i == 0) and self.local_global_model:
                    edge_index_in = edge_index_global
                    edge_attr_in = edge_attr_global
                else:
                    edge_index_in = edge_index_local
                    edge_attr_in = edge_attr_local
                        
            if self.use_norm:
                s, v = self.norms[i](x=(s, v), batch=batch)
            
            out = self.convs[i](x=(s, v, p), edge_index=edge_index_in, edge_attr=edge_attr_in)
            
            if self.fully_connected:
                s, v, p, e = out["s"], out["v"], out['p'], out["e"]
                p = p - scatter_mean(p, batch, dim=0)[batch]
                if self.recompute_edge_attributes:
                    edge_attr_global = self.calculate_edge_attrs(edge_index=edge_index_global, pos=p, edge_attr=e)
            else:
                if (i == self.num_layers - 2 or i == 0) and self.local_global_model:
                    s, v, p, e = out["s"], out["v"], out['p'], out["e"]
                    edge_global_dense[edge_index_global[0], edge_index_global[1], :] = e 
                    p = p - scatter_mean(p, batch, dim=0)[batch]
                    if self.recompute_edge_attributes:
                        edge_attr_global = self.calculate_edge_attrs(edge_index=edge_index_global, pos=p, edge_attr=e)
                    if self.recompute_radius_graph:
                        edge_index_local = radius_graph(p, r=self.cutoff_local, batch=batch, max_num_neighbors=128, flow='source_to_target')
                    edge_attr_local = edge_global_dense[edge_index_local[0], edge_index_local[1], :]
                    if self.recompute_edge_attributes:
                        edge_attr_local = self.calculate_edge_attrs(edge_index=edge_index_local, pos=p, edge_attr=edge_attr_local)
                else:
                    s, v, p, e = out["s"], out["v"], out["p"], out["e"]
                    edge_global_dense[edge_index_local[0], edge_index_local[1], :] = e
                    p = p - scatter_mean(p, batch, dim=0)[batch]
                    if self.recompute_radius_graph:
                        edge_index_local = radius_graph(p, r=self.cutoff_local, batch=batch, max_num_neighbors=128, flow='source_to_target')
                    edge_attr_local = edge_global_dense[edge_index_local[0], edge_index_local[1], :]
                    if self.recompute_edge_attributes:
                        edge_attr_local = self.calculate_edge_attrs(edge_index=edge_index_local, pos=p, edge_attr=edge_attr_local)
                    tmp_e = edge_global_dense[edge_index_global[0], edge_index_global[1], :]
                    if self.recompute_edge_attributes:
                        edge_attr_global = self.calculate_edge_attrs(edge_index=edge_index_global, pos=p, edge_attr=tmp_e)

            e = edge_attr_global[-1]       
             
        out = {"s": s, "v": v, "e": e, 'p': p}
        
        return out
    
    

class EQGATEdgeVirtualGNN(nn.Module):
    """_summary_
    EQGAT GNN Network updating node-level scalar, vectors and position features as well as edge-features.
    Args:
        nn (_type_): _description_
    """
    def __init__(self,
                 hn_dim: Tuple[int, int] = (64, 16),
                 rbf_dim: int = 64,
                 edge_dim: Optional[int] = 16,
                 cutoff_local: float = 5.0,
                 num_layers: int = 5,
                 use_norm: bool = True,
                 use_cross_product: bool = False,
                 vector_aggr: str = "mean",
                 fully_connected: bool = True,
                 local_global_model: bool = False,
                 recompute_radius_graph: bool = True,
                 recompute_edge_attributes: bool = True
                 ):
        super(EQGATEdgeVirtualGNN, self).__init__()

        assert fully_connected or local_global_model
        self.num_layers = num_layers
        self.fully_connected = fully_connected
        self.local_global_model = local_global_model
        self.cutoff_local = cutoff_local
        self.recompute_radius_graph = recompute_radius_graph
        self.recompute_edge_attributes = recompute_edge_attributes
        
        self.sdim, self.vdim = hn_dim
        self.edge_dim = edge_dim
        
        convs = []
        virtual_lins = []
        virtual_mlps = []
        
        for i in range(num_layers):
            if fully_connected:
                convs.append(
                    EQGATGlobalEdgeConvFinal(in_dims=hn_dim,
                                             out_dims=hn_dim,
                                             edge_dim=edge_dim,
                                             has_v_in=i>0,
                                             use_mlp_update= i < (num_layers - 1),
                                             vector_aggr=vector_aggr,
                                             use_cross_product=use_cross_product
                                             )
                )
            else:
                if (i == self.num_layers - 2 or i == 0) and local_global_model:
                    convs.append(
                        EQGATGlobalEdgeConvFinal(in_dims=hn_dim,
                                                 out_dims=hn_dim,
                                                 edge_dim=edge_dim,
                                                 has_v_in=i>0,
                                                 use_mlp_update= i < (num_layers - 1),
                                                 vector_aggr=vector_aggr,
                                                 use_cross_product=use_cross_product
                                                 )
                        )
                else:
                    convs.append(
                        EQGATLocalEdgeConvFinal(in_dims=hn_dim,
                                                out_dims=hn_dim,
                                                rbf_dim=rbf_dim,
                                                cutoff=cutoff_local,
                                                edge_dim=edge_dim,
                                                has_v_in=i>0,
                                                use_mlp_update= i < (num_layers - 1),
                                                vector_aggr=vector_aggr,
                                                use_cross_product=use_cross_product
                                                )
                    )
            
            virtual_lins.append(
               DenseLayer(hn_dim[0], hn_dim[0])
            )
            
            virtual_mlps.append(
                nn.Sequential(
                    DenseLayer(hn_dim[0], hn_dim[0], activation=nn.SiLU()),
                    DenseLayer(hn_dim[0], hn_dim[0]),
                )
            )
                    
        self.convs = nn.ModuleList(convs)
        self.virtual_lins = nn.ModuleList(virtual_lins)
        self.virtual_mlps = nn.ModuleList(virtual_mlps)
        
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
    
    def to_dense_edge_tensor(self, edge_index, edge_attr, num_nodes):
        E = torch.zeros(num_nodes, num_nodes, edge_attr.size(-1), device=edge_attr.device, dtype=edge_attr.dtype)
        E[edge_index[0], edge_index[1], :] = edge_attr
        return E
    
    def from_dense_edge_tensor(self, edge_index, E):
        return E[edge_index[0], edge_index[1], :]
    
    def forward(self,
                s: Tensor,
                v: Tensor,
                p: Tensor,
                edge_index_local: Tensor,
                edge_attr_local: Tuple[Tensor, Tensor, Tensor, Tensor],
                edge_index_global: Tensor,
                edge_attr_global: Tuple[Tensor, Tensor, Tensor, Tensor],
                batch: Tensor = None) -> Dict:
        
        # edge_attr_xyz (distances, cosines, relative_positions, edge_features)
        # (E, E, E x 3, E x F)
        n = s.size(0)
        bs = int(batch.max()) + 1
        
        edge_local_dense = torch.zeros(n, n, self.edge_dim, device=s.device, dtype=torch.float32)
        edge_global_dense = torch.zeros_like(edge_local_dense)
        edge_local_mask = torch.zeros(n, n, 1, device=s.device, dtype=torch.float32)
        edge_local_mask[edge_index_local[0], edge_index_local[1]] = 1.0
        
        g = torch.zeros_like(s)
        
        for i in range(len(self.convs)):   
            
            s = s + g
            
            if self.fully_connected:
                edge_index_in = edge_index_global
                edge_attr_in = edge_attr_global
            else:
                if (i == self.num_layers - 2 or i == 0) and self.local_global_model:
                    edge_index_in = edge_index_global
                    edge_attr_in = edge_attr_global
                else:
                    edge_index_in = edge_index_local
                    edge_attr_in = edge_attr_local
                        
            if self.use_norm:
                s, v = self.norms[i](x=(s, v), batch=batch)
            
            out = self.convs[i](x=(s, v, p), edge_index=edge_index_in, edge_attr=edge_attr_in)
            
            if self.fully_connected:
                s, v, p, e = out["s"], out["v"], out['p'], out["e"]
                p = p - scatter_mean(p, batch, dim=0)[batch]
                if self.recompute_edge_attributes:
                    edge_attr_global = self.calculate_edge_attrs(edge_index=edge_index_global, pos=p, edge_attr=e)
            else:
                if (i == self.num_layers - 2 or i == 0) and self.local_global_model:
                    s, v, p, e = out["s"], out["v"], out['p'], out["e"]
                    edge_global_dense[edge_index_global[0], edge_index_global[1], :] = e 
                    p = p - scatter_mean(p, batch, dim=0)[batch]
                    if self.recompute_edge_attributes:
                        edge_attr_global = self.calculate_edge_attrs(edge_index=edge_index_global, pos=p, edge_attr=e)
                    if self.recompute_radius_graph:
                        edge_index_local = radius_graph(p, r=self.cutoff_local, batch=batch, max_num_neighbors=128, flow='source_to_target')
                    edge_attr_local = edge_global_dense[edge_index_local[0], edge_index_local[1], :]
                    if self.recompute_edge_attributes:
                        edge_attr_local = self.calculate_edge_attrs(edge_index=edge_index_local, pos=p, edge_attr=edge_attr_local)
                else:
                    s, v, p, e = out["s"], out["v"], out["p"], out["e"]
                    edge_global_dense[edge_index_local[0], edge_index_local[1], :] = e
                    p = p - scatter_mean(p, batch, dim=0)[batch]
                    if self.recompute_radius_graph:
                        edge_index_local = radius_graph(p, r=self.cutoff_local, batch=batch, max_num_neighbors=128, flow='source_to_target')
                    edge_attr_local = edge_global_dense[edge_index_local[0], edge_index_local[1], :]
                    if self.recompute_edge_attributes:
                        edge_attr_local = self.calculate_edge_attrs(edge_index=edge_index_local, pos=p, edge_attr=edge_attr_local)
                    tmp_e = edge_global_dense[edge_index_global[0], edge_index_global[1], :]
                    if self.recompute_edge_attributes:
                        edge_attr_global = self.calculate_edge_attrs(edge_index=edge_index_global, pos=p, edge_attr=tmp_e)

            g = self.virtual_lins[i](s)
            g = scatter_add(g, index=batch, dim=0, dim_size=bs)
            g = self.virtual_mlps[i](g)
            g = g[batch]

            e = edge_attr_global[-1]       
        
        s = s + g
        
        out = {"s": s, "v": v, "e": e, 'p': p}
        
        return out