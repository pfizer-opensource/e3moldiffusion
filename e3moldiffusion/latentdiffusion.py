import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from torch_geometric.nn.inits import reset
from torch_scatter import scatter_mean
from torch_geometric.typing import OptTensor
from e3moldiffusion.modules import DenseLayer
from e3moldiffusion.gnn import TopoEdgeGNN, EQGATEdgeGNN
from e3moldiffusion.coordsatomsbonds import LatentEncoderNetwork, PredictionHeadEdge

Encoder = LatentEncoderNetwork

class Decoder(nn.Module):
    def __init__(self,
                 num_atom_features: int,
                 num_bond_types: int = 5,
                 hn_dim: Tuple[int, int] = (256, 64),
                 edge_dim: int = 32,
                 num_layers: int = 5,
                 latent_dim: Optional[int] = None,
                 use_cross_product: bool = False,
                 recompute_edge_attributes: bool = True,
                 vector_aggr: str = "mean"
                 ):
        super().__init__()
        
        self.sdim, self.vdim = hn_dim
        
        self.atom_mapping = DenseLayer(num_atom_features + latent_dim, hn_dim[0] + 3)
        self.bond_mapping = DenseLayer(num_bond_types, edge_dim)
        self.gnn = EQGATEdgeGNN(
            hn_dim=hn_dim,
            cutoff_local=None,
            edge_dim=edge_dim,
            latent_dim=None,
            num_layers=num_layers,
            use_cross_product=use_cross_product,
            vector_aggr=vector_aggr,
            fully_connected=True, 
            local_global_model=False,
            recompute_radius_graph=False,
            recompute_edge_attributes=recompute_edge_attributes,
            edge_mp=False
        )
        
        self.prediction_head = PredictionHeadEdge(hn_dim=hn_dim, 
                                            edge_dim=edge_dim, 
                                            num_atom_features=num_atom_features,
                                            num_bond_types=num_bond_types
                                            )
        
        self.reset_parameters()
        
    def reset_parameters(self):
        self.atom_mapping.reset_parameters()
        self.bond_mapping.reset_parameters()
        self.gnn.reset_parameters()
        self.prediction_head.reset_parameters()
        
    def calculate_edge_attrs(self, edge_index: Tensor, edge_attr: OptTensor, pos: Tensor, sqrt: bool = True):
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
        z: Tensor,
        rot: OptTensor,
        edge_index_global: Tensor,
        edge_attr_global: OptTensor = Tensor,
        batch: OptTensor = None
        ) -> Dict:
       
       
        if batch is None:
            batch = torch.zeros(x.size(0), device=x.device, dtype=torch.long)

        x = torch.concat([x, z], dim=-1)
        
        s, pos = self.atom_mapping(x).split([self.sdim, 3], dim=-1)
        if rot is not None:
            rot = torch.index_select(input=rot, dim=0, index=batch)
            pos = torch.einsum('bij, bj -> bi', rot, pos)
            
        edge_attr_global_transformed = self.bond_mapping(edge_attr_global)
        
        edge_attr_global_transformed = self.calculate_edge_attrs(edge_index=edge_index_global, edge_attr=edge_attr_global_transformed, pos=pos, sqrt=True)
        
        v = torch.zeros(size=(x.size(0), 3, self.vdim), device=s.device)

        out = self.gnn(
            s=s, v=v, p=pos, z=z,
            edge_index_local=None, edge_attr_local=(None, None, None),
            edge_index_global=edge_index_global, edge_attr_global=edge_attr_global_transformed,
            batch=batch
        )
        
        out = self.prediction_head(x=out, batch=batch, edge_index_global=edge_index_global)
  
        return out
        

class LatentPredictionHeadEdge(nn.Module):
    def __init__(self, in_dim: int, edge_dim: int, num_atom_features: int, latent_dim: int, num_bond_types: int = 5) -> None:
        super(LatentPredictionHeadEdge, self).__init__()
        self.in_dim = in_dim
        self.num_atom_features = num_atom_features
        self.latent_dim = latent_dim
        self.shared_mapping = DenseLayer(self.in_dim, self.in_dim, bias=True, activation=nn.SiLU())
        
        self.bond_mapping = DenseLayer(edge_dim, self.in_dim, bias=True)
        self.bonds_lin = DenseLayer(in_features=self.in_dim, out_features=num_bond_types, bias=True)
        self.atoms_lin = DenseLayer(in_features=self.in_dim, out_features=num_atom_features + latent_dim, bias=True)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        self.shared_mapping.reset_parameters()
        self.coords_lin.reset_parameters()
        self.atoms_lin.reset_parameters()
        self.bonds_lin.reset_parameters()
        
    def forward(self,
                x: Dict,
                batch: Tensor,
                edge_index_global: Tensor
                ) -> Dict:
        
        s, e = x["s"], x["e"]
        s = self.shared_mapping(s)
        j, i = edge_index_global
        n = s.size(0)
        
        atoms_pred, latent_pred = self.atoms_lin(s).split([self.num_atom_features, self.latent_dim], dim=-1)

        e_dense = torch.zeros(n, n, e.size(-1), device=e.device)
        e_dense[edge_index_global[0], edge_index_global[1], :] = e
        e_dense = 0.5 * (e_dense + e_dense.permute(1, 0, 2))
        e = e_dense[edge_index_global[0], edge_index_global[1], :]
        
        f = F.silu(s[i] + s[j] + self.bond_mapping(e))      
        bonds_pred = self.bonds_lin(f)

        out = {"latent_pred": latent_pred,
               "atoms_pred": atoms_pred,
               "bonds_pred": bonds_pred
               }
        
        return out
    
    
class DenoisingLatentEdgeNetwork(nn.Module):
    """_summary_
    Denoising network that inputs:
        atom features, edge features, position features
    The network is tasked for data prediction, i.e. x0 parameterization as commonly known in the literature:
        atom features, edge features, position features
    Args:
        nn (_type_): _description_
    """
    def __init__(self,
                 num_atom_features: int,
                 num_bond_types: int = 5,
                 in_dim: int = 128,
                 edge_dim: int = 32,
                 num_layers: int = 5,
                 latent_dim: Optional[int] = None,
                 atom_mapping: bool = True,
                 bond_mapping: bool = True,
                 ) -> None:
        super(DenoisingLatentEdgeNetwork, self).__init__()
        
        self.time_mapping_atom = DenseLayer(1, in_dim)
        self.time_mapping_bond = DenseLayer(1, edge_dim) 
        
        if atom_mapping:
            self.atom_mapping = DenseLayer(num_atom_features, in_dim)
        else:
            self.atom_mapping = nn.Identity()
        
        if bond_mapping:
            self.bond_mapping = DenseLayer(num_bond_types, edge_dim)
        else:
            self.bond_mapping = nn.Identity()
        
        self.latent_mapping = DenseLayer(latent_dim, in_dim)
        self.atom_time_mapping = DenseLayer(in_dim, in_dim)
        self.bond_time_mapping = DenseLayer(edge_dim, edge_dim)
               
        self.gnn = TopoEdgeGNN(
            in_dim=in_dim,
            edge_dim=edge_dim,
            num_layers=num_layers
        )
        
        self.prediction_head = LatentPredictionHeadEdge(in_dim=in_dim, 
                                                        latent_dim=latent_dim,
                                                        edge_dim=edge_dim, 
                                                        num_atom_features=num_atom_features,
                                                        num_bond_types=num_bond_types
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
         
    def forward(
        self,
        x: Tensor,
        t: Tensor,
        z: Tensor,
        edge_index: Tensor,
        edge_attr: OptTensor = Tensor,
        batch: OptTensor = None,
        batch_edge_global: OptTensor = None) -> Dict:
        
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
        z = self.latent_mapping(z)
        s = s + z
        
        edge_attr_global_transformed = self.bond_mapping(edge_attr)
        edge_attr_global_transformed = self.bond_time_mapping(edge_attr_global_transformed + tedge_global) 
        
        out = self.gnn(
            s=s,
            edge_index=edge_index, edge_attr=edge_attr_global_transformed,
            batch=batch
        )
        
        out = self.prediction_head(x=out, batch=batch, edge_index_global=edge_index)
        
        #out['coords_perturbed'] = pos
        #out['atoms_perturbed'] = x
        #out['bonds_perturbed'] = edge_attr_global
        
        return out
    

