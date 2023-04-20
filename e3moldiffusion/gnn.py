from typing import Tuple, Dict, Optional
from torch import Tensor, nn
from torch_geometric.typing import OptTensor

from e3moldiffusion.convs import EQGATRBFConv, EQGATConv
from e3moldiffusion.modules import LayerNorm



class EncoderGNN(nn.Module):
    def __init__(self,
                 hn_dim: Tuple[int, int] = (64, 16),
                 rbf_dim: int = 64,
                 edge_dim: Optional[int] = None,
                 cutoff_local: float = 5.0,
                 num_layers: int = 5,
                 use_norm: bool = True,
                 use_cross_product: bool = False,
                 vector_aggr: str = "mean",
                 fully_connected: bool = False,
                 local_global_model: bool = True
                 ):
        super(EncoderGNN, self).__init__()

        self.num_layers = num_layers
        self.fully_connected = fully_connected
        self.local_global_model = local_global_model
        
        self.sdim, self.vdim = hn_dim
        
        convs = []
        for i in range(num_layers):
            if fully_connected:
                convs.append(
                    EQGATConv(in_dims=hn_dim,
                              out_dims=hn_dim,
                              edge_dim=edge_dim,
                              has_v_in=i>0,
                              use_mlp_update= i < (num_layers - 1),
                              vector_aggr=vector_aggr,
                              use_cross_product=use_cross_product
                              )
                )
            else:
                if i == self.num_layers - 2 and local_global_model:
                    convs.append(
                        EQGATConv(in_dims=hn_dim,
                                  out_dims=hn_dim,
                                  edge_dim=edge_dim,
                                  has_v_in=i>0,
                                  use_mlp_update= i < (num_layers - 1),
                                  vector_aggr=vector_aggr,
                                  use_cross_product=use_cross_product)
                        )
                else:
                    convs.append(
                        EQGATRBFConv(in_dims=hn_dim,
                         out_dims=hn_dim,
                         rbf_dim=rbf_dim,
                         edge_dim=edge_dim,
                         cutoff=cutoff_local,
                         use_cutoff_fnc=True,
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
                
    def forward(self,
                s: Tensor,
                v: Tensor,
                edge_index_local: Tensor,
                edge_attr_local: Tuple[Tensor, Tensor, OptTensor],
                edge_index_global: Tensor,
                edge_attr_global: Tuple[Tensor, Tensor, OptTensor],
                batch: Tensor = None) -> Dict:
        
        for i in range(len(self.convs)):    
            if self.local_global_model:
                if i == self.num_layers - 2:
                    edge_index_in = edge_index_global
                    edge_attr_in = edge_attr_global
                else:
                    edge_index_in = edge_index_local
                    edge_attr_in = edge_attr_local
            else:
                if self.fully_connected:
                    edge_index_in = edge_index_global
                    edge_attr_in = edge_attr_global
                else:
                    edge_index_in = edge_index_local
                    edge_attr_in = edge_attr_local

            if self.use_norm:
                s, v = self.norms[i](x=(s, v), batch=batch)
                
            s, v = self.convs[i](x=(s, v), edge_index=edge_index_in, edge_attr=edge_attr_in)
                        
        out = {"s": s, "v": v}
        
        return out

class EncoderGNNAtomBond(nn.Module):
    def __init__(self,
                 hn_dim: Tuple[int, int] = (64, 16),
                 rbf_dim: int = 64,
                 edge_dim: Optional[int] = None,
                 cutoff_local: float = 5.0,
                 num_layers: int = 5,
                 use_norm: bool = True,
                 use_cross_product: bool = False,
                 vector_aggr: str = "mean",
                 fully_connected: bool = False,
                 local_global_model: bool = True,
                 local_edge_attrs: bool = False
                 ):
        super(EncoderGNNAtomBond, self).__init__()

        self.num_layers = num_layers
        self.fully_connected = fully_connected
        self.local_global_model = local_global_model
        
        self.sdim, self.vdim = hn_dim
        
        convs = []
        
        for i in range(num_layers):
            if fully_connected:
                convs.append(
                    EQGATConv(in_dims=hn_dim,
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
                        EQGATConv(in_dims=hn_dim,
                                  out_dims=hn_dim,
                                  edge_dim=edge_dim,
                                  has_v_in=i>0,
                                  use_mlp_update= i < (num_layers - 1),
                                  vector_aggr=vector_aggr,
                                  use_cross_product=use_cross_product)
                        )
                else:
                    convs.append(
                        EQGATRBFConv(in_dims=hn_dim,
                         out_dims=hn_dim,
                         rbf_dim=rbf_dim,
                         edge_dim=edge_dim if local_edge_attrs else None,
                         cutoff=cutoff_local,
                         use_cutoff_fnc=True,
                         has_v_in=i>0,
                         use_mlp_update= i < (num_layers - 1),
                         vector_aggr=vector_aggr,
                         use_cross_product=use_cross_product
                         )
                    )
    
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
                
    def forward(self,
                s: Tensor,
                v: Tensor,
                edge_index_local: Tensor,
                edge_attr_local: Tuple[Tensor, Tensor, OptTensor],
                edge_index_global: Tensor,
                edge_attr_global: Tuple[Tensor, Tensor, OptTensor],
                batch: Tensor = None) -> Dict:
        
        for i in range(len(self.convs)):    
            if self.local_global_model:
                if i == self.num_layers - 2 or i == 0:
                    edge_index_in = edge_index_global
                    edge_attr_in = edge_attr_global
                else:
                    edge_index_in = edge_index_local
                    edge_attr_in = edge_attr_local
            else:
                if self.fully_connected:
                    edge_index_in = edge_index_global
                    edge_attr_in = edge_attr_global
                else:
                    edge_index_in = edge_index_local
                    edge_attr_in = edge_attr_local

            if self.use_norm:
                s, v = self.norms[i](x=(s, v), batch=batch)
                
            s, v = self.convs[i](x=(s, v), edge_index=edge_index_in, edge_attr=edge_attr_in)
                        
        out = {"s": s, "v": v}
        
        return out

