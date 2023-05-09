import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict

from e3moldiffusion.gnn import EncoderGNNAtomBond
from e3moldiffusion.modules import DenseLayer
from torch_geometric.typing import OptTensor

from torch import Tensor, nn
from torch_scatter import scatter_add
from torch_geometric.nn.inits import reset

# Score Model that is trained to learn 3D coords, Atom-Types and Bond-Types


class ScoreHead(nn.Module):
    def __init__(self, hn_dim: Tuple[int, int], num_atom_types: int, num_bond_types: int = 5) -> None:
        super(ScoreHead, self).__init__()
        self.sdim, self.vdim = hn_dim
        self.num_atom_types = num_atom_types
        
        self.bonds_lin_0 = DenseLayer(in_features=self.sdim, out_features=self.sdim, bias=False)
        self.bonds_lin_1 = DenseLayer(in_features=self.sdim, out_features=num_bond_types, bias=False)
        self.coords_lin = DenseLayer(in_features=self.vdim, out_features=1, bias=False)
        self.atoms_lin = DenseLayer(in_features=self.sdim, out_features=num_atom_types, bias=False)
        self.reset_parameters()
        
    def reset_parameters(self):
        self.coords_lin.reset_parameters()
        self.atoms_lin.reset_parameters()
        self.bonds_lin_0.reset_parameters()
        self.bonds_lin_1.reset_parameters()
        
    def forward(self,
                x: Dict[Tensor, Tensor],
                pos: Tensor,
                batch: Tensor,
                edge_index_global: Tensor
                ) -> Dict:
        
        s, v, e = x["s"], x["v"], x["e"]
        
        j, i = edge_index_global
        eps_ij = F.silu(self.bonds_lin_0(s[i] + s[j] + e))
        eps_ij = self.bonds_lin_1(eps_ij)
        
        score_coords = self.coords_lin(v).squeeze()
        score_atoms = self.atoms_lin(s)
        
        out = {"score_coords": score_coords, "score_atoms": score_atoms, "score_bonds": eps_ij}
        
        return out


class ScoreModel(nn.Module):
    def __init__(self,
                 num_atom_types: int,
                 num_bond_types: int = 5,
                 hn_dim: Tuple[int, int] = (64, 16),
                 rbf_dim: int = 16,
                 cutoff_local: float = 5.0,
                 num_layers: int = 5,
                 use_norm: bool = True,
                 use_cross_product: bool = False,
                 fully_connected: bool = False,
                 local_global_model: bool = True,
                 local_edge_attrs: bool = False,
                 vector_aggr: str = "mean"
                 ) -> None:
        super(ScoreModel, self).__init__()
        
        self.time_mapping = DenseLayer(1, hn_dim[0])   
        self.atom_mapping = DenseLayer(num_atom_types, hn_dim[0])
        self.bond_mapping = DenseLayer(num_bond_types, hn_dim[0])
        
        self.atom_time_mapping = DenseLayer(hn_dim[0], hn_dim[0])
        self.bond_time_mapping = DenseLayer(hn_dim[0], hn_dim[0])
        
        
        self.sdim, self.vdim = hn_dim
        
        self.local_global_model = local_global_model
        self.fully_connected = fully_connected
        
        assert fully_connected or local_global_model
        
        self.gnn = EncoderGNNAtomBond(
            hn_dim=hn_dim,
            cutoff_local=cutoff_local,
            rbf_dim=rbf_dim,
            edge_dim=hn_dim[0],
            num_layers=num_layers,
            use_norm=use_norm,
            use_cross_product=use_cross_product,
            vector_aggr=vector_aggr,
            fully_connected=fully_connected, 
            local_global_model=local_global_model,
            local_edge_attrs=False #local_edge_attrs
        )
        
        self.score_head = ScoreHead(hn_dim=hn_dim, num_atom_types=num_atom_types, num_bond_types=num_bond_types)
        
        self.reset_parameters()

    def reset_parameters(self):
        self.atom_mapping.reset_parameters()
        self.bond_mapping.reset_parameters()
        self.time_mapping.reset_parameters()
        self.atom_time_mapping.reset_parameters()
        self.bond_time_mapping.reset_parameters()
        self.gnn.reset_parameters()
        self.score_head.reset_parameters()
        
    def calculate_edge_attrs(self, edge_index: Tensor, edge_attr: OptTensor, pos: Tensor):
        source, target = edge_index
        r = pos[target] - pos[source]
        d = torch.clamp(torch.pow(r, 2).sum(-1), min=1e-6).sqrt()
        r_norm = torch.div(r, d.unsqueeze(-1))
        edge_attr = (d, r_norm, edge_attr)
        return edge_attr
    
    def forward(
        self,
        x: Tensor,
        t: Tensor,
        pos: Tensor,
        edge_index_local: Tensor,
        edge_index_global: Tensor,
        edge_attr_local: OptTensor = None,
        edge_attr_global: OptTensor = None,
        batch: OptTensor = None,
        batch_edge_local: OptTensor = None,
        batch_edge_global: OptTensor = None) -> Dict[Tensor, Tensor]:
        
        # t: (batch_size,)
        t = self.time_mapping(t)
        tnode = t[batch]
        
        # edge_index_global (2, E*)
        tedge_global = t[batch_edge_global]
        tedge_local = t[batch_edge_local]
        
        if batch is None:
            batch = torch.zeros(x.size(0), device=x.device, dtype=torch.long)
     
        s = self.atom_mapping(x)
        # s = self.atom_time_mapping(F.silu(s + tnode))  # not use activation
        s = self.atom_time_mapping(s + tnode)
        
        edge_attr_global = self.bond_mapping(edge_attr_global)
        # edge_attr_global = self.bond_time_mapping(F.silu(edge_attr_global + tedge_global))  # not use activation
        edge_attr_global = self.bond_time_mapping(edge_attr_global + tedge_global)
        
        #edge_attr_local = self.bond_mapping(edge_attr_local)
        #edge_attr_local = self.bond_time_mapping(F.silu(edge_attr_local + tedge_local))
        #edge_attr_local = self.bond_time_mapping(edge_attr_local + tedge_local)
        edge_attr_local = None
        
        # local
        edge_attr_local = self.calculate_edge_attrs(edge_index=edge_index_local, edge_attr=edge_attr_local, pos=pos)        
        # global
        if self.local_global_model or self.fully_connected:
            edge_attr_global = self.calculate_edge_attrs(edge_index=edge_index_global, edge_attr=edge_attr_global, pos=pos)
        else:
            edge_attr_global = (None, None, None)
        
        
        v = torch.zeros(size=(x.size(0), 3, self.vdim), device=s.device)

        out = self.gnn(
            s=s, v=v,
            edge_index_local=edge_index_local, edge_attr_local=edge_attr_local,
            edge_index_global=edge_index_global, edge_attr_global=edge_attr_global,
            batch=batch
        )
        
        score = self.score_head(x=out, pos=pos, batch=batch, edge_index_global=edge_index_global)
             
        return score
    
        
        
if __name__ == "__main__":
    pass