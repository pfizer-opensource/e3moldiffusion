from typing import Dict, Tuple, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.typing import OptTensor
from torch_scatter import scatter_mean, scatter_add

from e3moldiffusion.gnn import EQGATGNN, EQGATEdgeGNN
from e3moldiffusion.coordsatomsbonds import DenoisingNetwork
from e3moldiffusion.modules import DenseLayer
from torch_geometric.utils import softmax
from torch_geometric.nn.inits import reset

    
class SoftMaxAttentionAggregation(nn.Module):
    """
    Softmax Attention Pooling as proposed "Graph Matching Networks
    for Learning the Similarity of Graph Structured Objects"
    <https://arxiv.org/abs/1904.12787>
    """

    def __init__(
        self,
        dim: int
    ):
        super(SoftMaxAttentionAggregation, self).__init__()

        self.node_net = nn.Sequential(
            DenseLayer(dim, dim, activation=nn.SiLU()),
            DenseLayer(dim, dim)
        )
        self.gate_net = nn.Sequential(
            DenseLayer(dim, dim, activation=nn.SiLU()),
            DenseLayer(dim, 1)
        )
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.node_net)
        reset(self.gate_net)

    def forward(self, x: Tensor, index: Optional[Tensor] = None,  dim_size: Optional[int] = None,
                dim: int = -2) -> Tensor:

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
    
class EncoderEdgeGNN(nn.Module):
    def __init__(self,
                 num_atom_types: int,
                 num_bond_types: int = 5,
                 hn_dim: Tuple[int, int] = (256, 64),
                 latent_dim: int = 128,
                 rbf_dim: int = 32,
                 edge_dim: int = 32,
                 cutoff_local: float = 7.5,
                 num_layers: int = 5,
                 use_norm: bool = True,
                 use_cross_product: bool = False,
                 fully_connected: bool = True,
                 local_global_model: bool = False,
                 recompute_radius_graph: bool = False,
                 recompute_edge_attributes: bool = False,
                 vector_aggr: str = "mean",
                 ) -> None:
        super(EncoderEdgeGNN, self).__init__()
      
        self.atom_mapping = DenseLayer(num_atom_types, hn_dim[0])
        self.bond_mapping = DenseLayer(num_bond_types, edge_dim)
      
        assert fully_connected or local_global_model
        
        self.sdim, self.vdim = hn_dim
        
        self.local_global_model = local_global_model
        self.fully_connected = fully_connected
        
        assert fully_connected or local_global_model
    
        self.gnn = EQGATEdgeGNN(
            hn_dim=hn_dim,
            cutoff_local=cutoff_local,
            rbf_dim=rbf_dim,
            edge_dim=edge_dim,
            num_layers=num_layers,
            use_norm=use_norm,
            use_cross_product=use_cross_product,
            vector_aggr=vector_aggr,
            fully_connected=fully_connected, 
            local_global_model=local_global_model,
            recompute_radius_graph=recompute_radius_graph,
            recompute_edge_attributes=recompute_edge_attributes
        )
        
        self.latent_mapping = DenseLayer(hn_dim[0], latent_dim)
        self.graph_pooling = SoftMaxAttentionAggregation(dim=latent_dim)
        
        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self.atom_mapping, "reset_parameters"):
            self.atom_mapping.reset_parameters()
        if hasattr(self.bond_mapping, "reset_parameters"):
            self.bond_mapping.reset_parameters()
        self.gnn.reset_parameters()
        self.latent_mapping.reset_parameters()
        self.graph_pooling.reset_parameters()
        
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
        pos: Tensor,
        edge_index_local: Tensor,
        edge_index_global: Tensor,
        edge_attr_global: OptTensor = Tensor,
        batch: OptTensor = None) -> Dict:
        
        pos = pos - scatter_mean(pos, index=batch, dim=0)[batch]
      
        if batch is None:
            batch = torch.zeros(x.size(0), device=x.device, dtype=torch.long)
     
        s = self.atom_mapping(x)

        edge_attr_global_transformed = self.bond_mapping(edge_attr_global)        
        edge_dense = torch.zeros(x.size(0), x.size(0), edge_attr_global_transformed.size(-1), device=s.device)
        edge_dense[edge_index_global[0], edge_index_global[1], :] = edge_attr_global_transformed
        
        edge_attr_local_transformed = edge_dense[edge_index_local[0], edge_index_local[1], :]
        
        # local
        edge_attr_local_transformed = self.calculate_edge_attrs(edge_index=edge_index_local, edge_attr=edge_attr_local_transformed, pos=pos)        
        # global
        edge_attr_global_transformed = self.calculate_edge_attrs(edge_index=edge_index_global, edge_attr=edge_attr_global_transformed, pos=pos)
        
        v = torch.zeros(size=(x.size(0), 3, self.vdim), device=s.device)

        out = self.gnn(
            s=s, v=v, p=pos,
            edge_index_local=edge_index_local, edge_attr_local=edge_attr_local_transformed,
            edge_index_global=edge_index_global, edge_attr_global=edge_attr_global_transformed,
            batch=batch
        )
        
        out = self.latent_mapping(out["s"])
        out = {"graph_embedding": self.graph_pooling(x=out, index=batch, dim=0)}
        
        return out
    

class EncoderGNN(nn.Module):
    def __init__(self,
                 num_atom_types: int,
                 hn_dim: Tuple[int, int] = (256, 64),
                 rbf_dim: int = 32,
                 latent_dim: int = 128,
                 cutoff_local: float = 7.5,
                 num_layers: int = 5,
                 use_norm: bool = True,
                 use_cross_product: bool = False,
                 fully_connected: bool = False,
                 local_global_model: bool = True,
                 recompute_radius_graph: bool = False,
                 recompute_edge_attributes: bool = False,
                 vector_aggr: str = "mean",
                 ) -> None:
        super(EncoderGNN, self).__init__()
        

        self.atom_mapping = DenseLayer(num_atom_types, hn_dim[0])
        assert fully_connected or local_global_model
        self.sdim, self.vdim = hn_dim
        
        self.local_global_model = local_global_model
        self.fully_connected = fully_connected
        
        assert fully_connected or local_global_model
    
        self.gnn = EQGATGNN(
            hn_dim=hn_dim,
            cutoff_local=cutoff_local,
            rbf_dim=rbf_dim,
            num_layers=num_layers,
            use_norm=use_norm,
            use_cross_product=use_cross_product,
            vector_aggr=vector_aggr,
            fully_connected=fully_connected, 
            local_global_model=local_global_model,
            recompute_radius_graph=recompute_radius_graph,
            recompute_edge_attributes=recompute_edge_attributes
        )
        
        self.latent_mapping = DenseLayer(hn_dim[0], latent_dim)
        self.graph_pooling = SoftMaxAttentionAggregation(dim=hn_dim[0])
        
        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self.atom_mapping, "reset_parameters"):
            self.atom_mapping.reset_parameters()
        self.gnn.reset_parameters()
        self.latent_mapping.reset_parameters()
        self.graph_pooling.reset_parameters()
        
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
        pos: Tensor,
        edge_index_local: Tensor,
        edge_index_global: Tensor,
        batch: OptTensor = None) -> Dict:
        
        pos = pos - scatter_mean(pos, index=batch, dim=0)[batch]
        if batch is None:
            batch = torch.zeros(x.size(0), device=x.device, dtype=torch.long)
     
        s = self.atom_mapping(x)
        v = torch.zeros(size=(x.size(0), 3, self.vdim), device=s.device)
        out = self.gnn(
            s=s, v=v, p=pos,
            edge_index_local=edge_index_local, edge_attr_local=None,
            edge_index_global=edge_index_global, edge_attr_global=None,
            batch=batch
        )
        
        out = self.latent_mapping(out["s"])
        out = {"graph_embedding": self.graph_pooling(x=out, index=batch, dim=0)}
        
        return out
    
class DenoisingDecoderNetwork(DenoisingNetwork):
    def __init__(self,
                 num_atom_types: int,
                 num_bond_types: int = 5,
                 hn_dim: Tuple[int, int] = (256, 64),
                 rbf_dim: int = 32,
                 cutoff_local: float = 7.5,
                 num_layers: int = 5,
                 use_norm: bool = True,
                 use_cross_product: bool = False,
                 fully_connected: bool = False,
                 local_global_model: bool = True,
                 recompute_radius_graph: bool = True,
                 recompute_edge_attributes: bool = True,
                 vector_aggr: str = "mean",
                 ) -> None:
        super().__init__(num_atom_types,
                         num_bond_types,
                         hn_dim, 
                         rbf_dim, 
                         cutoff_local,
                         num_layers, 
                         use_norm,
                         use_cross_product, 
                         fully_connected, 
                         local_global_model, 
                         recompute_radius_graph, 
                         recompute_edge_attributes, 
                         vector_aggr,
                         False)
 
if __name__ == "__main__":
    pass