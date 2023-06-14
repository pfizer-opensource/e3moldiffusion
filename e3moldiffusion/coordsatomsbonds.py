import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict

from e3moldiffusion.gnn import EncoderGNNAtomBond, EncoderGNNAtomBondV2, EncoderGNNAtomBondSE3, GNNSE3AtomBond, GNNSE3Atom, GNNSE3AtomBondFinal
from e3moldiffusion.modules import DenseLayer
from torch_geometric.typing import OptTensor

from torch import Tensor, nn
from torch_scatter import scatter_add, scatter_mean
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
        
        self.local_edge_attrs = local_edge_attrs
        self.local_global_model = local_global_model
        self.fully_connected = fully_connected
        
        assert fully_connected or local_global_model
        
        if not local_edge_attrs:
            self.gnn = EncoderGNNAtomBond(
                hn_dim=hn_dim,
                cutoff_local=cutoff_local,
                rbf_dim=rbf_dim,
                edge_dim=hn_dim[0],  # we should experiment and reduce the edge_dim size
                num_layers=num_layers,
                use_norm=use_norm,
                use_cross_product=use_cross_product,
                vector_aggr=vector_aggr,
                fully_connected=fully_connected, 
                local_global_model=local_global_model
            )
        else:
            self.gnn = EncoderGNNAtomBondV2(
                hn_dim=hn_dim,
                cutoff_local=cutoff_local,
                rbf_dim=rbf_dim,
                edge_dim=hn_dim[0], # we should experiment and reduce the edge_dim size
                num_layers=num_layers,
                use_norm=use_norm,
                use_cross_product=use_cross_product,
                vector_aggr=vector_aggr,
                fully_connected=fully_connected, 
                local_global_model=local_global_model
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
        s = self.atom_time_mapping(s + tnode)
        
        edge_attr_global_transformed = self.bond_mapping(edge_attr_global)
        edge_attr_global_transformed = self.bond_time_mapping(edge_attr_global_transformed + tedge_global)
        
        if self.local_edge_attrs:
            edge_attr_local = self.bond_mapping(edge_attr_local)
            edge_attr_local = self.bond_time_mapping(edge_attr_local + tedge_local)
        else:
            edge_attr_local = None
        
        # local
        edge_attr_local = self.calculate_edge_attrs(edge_index=edge_index_local, edge_attr=edge_attr_local, pos=pos)        
        # global
        if self.local_global_model or self.fully_connected:
            edge_attr_global_transformed = self.calculate_edge_attrs(edge_index=edge_index_global, edge_attr=edge_attr_global_transformed, pos=pos)
        else:
            edge_attr_global_transformed = (None, None, None)
        
        
        v = torch.zeros(size=(x.size(0), 3, self.vdim), device=s.device)

        out = self.gnn(
            s=s, v=v,
            edge_index_local=edge_index_local, edge_attr_local=edge_attr_local,
            edge_index_global=edge_index_global, edge_attr_global=edge_attr_global_transformed,
            batch=batch
        )
        
        out = self.score_head(x=out, pos=pos, batch=batch, edge_index_global=edge_index_global)
        
        out['coords_perturbed'] = pos
        out['atoms_perturbed'] = x
        out['bonds_perturbed'] = edge_attr_global
        

        return out
    
    
class PredictionHead(nn.Module):
    def __init__(self, hn_dim: Tuple[int, int], edge_dim: int, num_atom_types: int, num_bond_types: int = 5) -> None:
        super(PredictionHead, self).__init__()
        self.sdim, self.vdim = hn_dim
        self.num_atom_types = num_atom_types
        
        self.shared_mapping = DenseLayer(self.sdim, self.sdim, bias=True, activation=nn.SiLU())
        
        self.bond_mapping = DenseLayer(edge_dim, self.sdim, bias=True)
        
        self.bonds_lin_0 = DenseLayer(in_features=self.sdim + 1, out_features=self.sdim, bias=True)
        self.bonds_lin_1 = DenseLayer(in_features=self.sdim, out_features=2 * num_bond_types, bias=True)
        self.coords_lin = DenseLayer(in_features=self.vdim, out_features=2 * 1, bias=False)
        self.atoms_lin = DenseLayer(in_features=self.sdim, out_features=2 * num_atom_types, bias=True)
        self.reset_parameters()
        
    def reset_parameters(self):
        self.shared_mapping.reset_parameters()
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
        s = self.shared_mapping(s)
        j, i = edge_index_global
        
        d = (pos[i] - pos[j]).pow(2).sum(-1, keepdim=True).sqrt()
        f = s[i] + s[j] + self.bond_mapping(e)
        edge = torch.cat([f, d], dim=-1)
        
        bonds_pred = F.silu(self.bonds_lin_0(edge))
        bonds_pred = self.bonds_lin_1(bonds_pred)
        bonds_pred, bonds_eps = bonds_pred.chunk(2, dim=-1)
        
        coords_eps_0 = self.coords_lin(v)
        coords_eps_0 = coords_eps_0 - scatter_mean(coords_eps_0, index=batch, dim=0)[batch]

        coords_eps_0, coords_eps_1 = coords_eps_0.chunk(2, dim=-1)
        coords_eps_0, coords_eps_1 = coords_eps_0.squeeze(), coords_eps_1.squeeze()
        
        atoms_eps, atoms_pred = self.atoms_lin(s).chunk(2, dim=-1)
        
        pos = pos - scatter_mean(pos, index=batch, dim=0)[batch]
        coords_pred = pos + coords_eps_1
        
        out = {"coords_pred": coords_pred, "coords_eps": coords_eps_0,
               "atoms_pred": atoms_pred, "atoms_eps": atoms_eps,
               "bonds_pred": bonds_pred, "bonds_eps": bonds_eps
               }
        
        return out

 
    
class PredictionHeadFinal(nn.Module):
    def __init__(self, hn_dim: Tuple[int, int], edge_dim: int, num_atom_types: int, num_bond_types: int = 5) -> None:
        super(PredictionHeadFinal, self).__init__()
        self.sdim, self.vdim = hn_dim
        self.num_atom_types = num_atom_types
        
        self.shared_mapping = DenseLayer(self.sdim, self.sdim, bias=True, activation=nn.SiLU())
        
        self.bond_mapping = DenseLayer(edge_dim, self.sdim, bias=True)
        
        self.bonds_lin_0 = DenseLayer(in_features=self.sdim + 1, out_features=self.sdim, bias=True)
        self.bonds_lin_1 = DenseLayer(in_features=self.sdim, out_features=num_bond_types, bias=True)
        self.coords_lin = DenseLayer(in_features=self.vdim, out_features=1, bias=False)
        self.atoms_lin = DenseLayer(in_features=self.sdim, out_features=num_atom_types, bias=True)
        self.reset_parameters()
        
    def reset_parameters(self):
        self.shared_mapping.reset_parameters()
        self.coords_lin.reset_parameters()
        self.atoms_lin.reset_parameters()
        self.bonds_lin_0.reset_parameters()
        self.bonds_lin_1.reset_parameters()
        
    def forward(self,
                x: Dict,
                batch: Tensor,
                edge_index_global: Tensor
                ) -> Dict:
        
        s, v, p, e = x["s"], x["v"], x['p'], x["e"]
        s = self.shared_mapping(s)
        j, i = edge_index_global
         
        coords_pred = self.coords_lin(v).squeeze()
        coords_pred = coords_pred - scatter_mean(coords_pred, index=batch, dim=0)[batch]

        atoms_pred = self.atoms_lin(s)
        
        p = p - scatter_mean(p, index=batch, dim=0)[batch]
        coords_pred = p + coords_pred
        
        d = (coords_pred[i] - coords_pred[j]).pow(2).sum(-1, keepdim=True).sqrt()
        f = s[i] + s[j] + self.bond_mapping(e)
        edge = torch.cat([f, d], dim=-1)
        
        bonds_pred = F.silu(self.bonds_lin_0(edge))
        bonds_pred = self.bonds_lin_1(bonds_pred)
        
        out = {"coords_pred": coords_pred,
               "atoms_pred": atoms_pred,
               "bonds_pred": bonds_pred
               }
        
        return out

class ScoreModelSE3(nn.Module):
    def __init__(self,
                 num_atom_types: int,
                 num_bond_types: int = 5,
                 hn_dim: Tuple[int, int] = (256, 64),
                 rbf_dim: int = 32,
                 edge_dim: int = 32,
                 cutoff_local: float = 7.5,
                 num_layers: int = 5,
                 use_norm: bool = True,
                 use_cross_product: bool = False,
                 fully_connected: bool = False,
                 local_global_model: bool = True,
                 vector_aggr: str = "mean",
                 atom_mapping: bool = True,
                 bond_mapping: bool = True,
                 ) -> None:
        super(ScoreModelSE3, self).__init__()
        
        self.time_mapping_atom = DenseLayer(1, hn_dim[0])
        self.time_mapping_bond = DenseLayer(1, edge_dim) 
        
        if atom_mapping:
            self.atom_mapping = DenseLayer(num_atom_types, hn_dim[0])
        else:
            self.atom_mapping = nn.Identity()
        
        if bond_mapping:
            self.bond_mapping = DenseLayer(num_bond_types, edge_dim)
        else:
            self.bond_mapping = nn.Identity()
        
        self.atom_time_mapping = DenseLayer(hn_dim[0], hn_dim[0])
        self.bond_time_mapping = DenseLayer(edge_dim, edge_dim)
        
        assert fully_connected or local_global_model
        
        
        self.sdim, self.vdim = hn_dim
        
        self.local_global_model = local_global_model
        self.fully_connected = fully_connected
        
        assert fully_connected or local_global_model
    
        self.gnn = GNNSE3AtomBondFinal(
            hn_dim=hn_dim,
            cutoff_local=cutoff_local,
            rbf_dim=rbf_dim,
            edge_dim=edge_dim,
            num_layers=num_layers,
            use_norm=use_norm,
            use_cross_product=use_cross_product,
            vector_aggr=vector_aggr,
            fully_connected=fully_connected, 
            local_global_model=local_global_model
        )
        
        self.prediction_head = PredictionHeadFinal(hn_dim=hn_dim, 
                                                   edge_dim=edge_dim, 
                                                   num_atom_types=num_atom_types,
                                                   num_bond_types=num_bond_types)
        
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
        pos: Tensor,
        edge_index_local: Tensor,
        edge_index_global: Tensor,
        edge_attr_global: OptTensor = Tensor,
        batch: OptTensor = None,
        batch_edge_global: OptTensor = None) -> Dict:
        
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
        
        edge_attr_global_transformed = self.bond_mapping(edge_attr_global)
        edge_attr_global_transformed = self.bond_time_mapping(edge_attr_global_transformed + tedge_global)
        
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
        
        out = self.prediction_head(x=out, batch=batch, edge_index_global=edge_index_global)
        
        out['coords_perturbed'] = pos
        out['atoms_perturbed'] = x
        out['bonds_perturbed'] = edge_attr_global
        
        return out
    
# Inputs: Coords, Atoms
# Outouts: Coords, Atoms, Edges


class PredictionHeadNew(nn.Module):
    def __init__(self, hn_dim: Tuple[int, int], num_atom_types: int, num_bond_types: int = 5) -> None:
        super(PredictionHeadNew, self).__init__()
        self.sdim, self.vdim = hn_dim
        self.num_atom_types = num_atom_types
        
        self.shared_mapping = DenseLayer(self.sdim, self.sdim, bias=True, activation=nn.SiLU())
        
        self.bonds_lin_0 = DenseLayer(in_features=self.sdim + 1, out_features=self.sdim, bias=True)
        self.bonds_lin_1 = DenseLayer(in_features=self.sdim, out_features=2 * num_bond_types, bias=True)
        self.coords_lin = DenseLayer(in_features=self.vdim, out_features=2 * 1, bias=False)
        self.atoms_lin = DenseLayer(in_features=self.sdim, out_features=2 * num_atom_types, bias=True)
        self.reset_parameters()
        
    def reset_parameters(self):
        self.shared_mapping.reset_parameters()
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
        s = self.shared_mapping(s)
        j, i = edge_index_global
        
        d = (pos[i] - pos[j]).pow(2).sum(-1, keepdim=True).sqrt()
        f = s[i] + s[j]
        edge = torch.cat([f, d], dim=-1)
        
        bonds_pred = F.silu(self.bonds_lin_0(edge))
        bonds_pred = self.bonds_lin_1(bonds_pred)
        bonds_pred, bonds_eps = bonds_pred.chunk(2, dim=-1)
        
        coords_eps_0 = self.coords_lin(v)
        coords_eps_0 = coords_eps_0 - scatter_mean(coords_eps_0, index=batch, dim=0)[batch]

        coords_eps_0, coords_eps_1 = coords_eps_0.chunk(2, dim=-1)
        coords_eps_0, coords_eps_1 = coords_eps_0.squeeze(), coords_eps_1.squeeze()
        
        atoms_eps, atoms_pred = self.atoms_lin(s).chunk(2, dim=-1)
        
        pos = pos - scatter_mean(pos, index=batch, dim=0)[batch]
        coords_pred = pos + coords_eps_1
        
        out = {"coords_pred": coords_pred, "coords_eps": coords_eps_0,
               "atoms_pred": atoms_pred, "atoms_eps": atoms_eps,
               "bonds_pred": bonds_pred, "bonds_eps": bonds_eps
               }
        
        return out


class ScoreModelSE3New(nn.Module):
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
                 vector_aggr: str = "mean"
                 ) -> None:
        super(ScoreModelSE3New, self).__init__()
        
        self.time_mapping = DenseLayer(1, hn_dim[0])   
        self.atom_mapping = DenseLayer(num_atom_types, hn_dim[0])
        self.atom_time_mapping = DenseLayer(hn_dim[0], hn_dim[0])
        
        assert fully_connected or local_global_model
        
        self.sdim, self.vdim = hn_dim
        
        self.local_global_model = local_global_model
        self.fully_connected = fully_connected
        
        assert fully_connected or local_global_model
    
        self.gnn = GNNSE3Atom(
            hn_dim=hn_dim,
            cutoff_local=cutoff_local,
            rbf_dim=rbf_dim,
            num_layers=num_layers,
            use_norm=use_norm,
            use_cross_product=use_cross_product,
            vector_aggr=vector_aggr,
            fully_connected=fully_connected, 
            local_global_model=local_global_model
        )
        
        self.score_head = PredictionHeadNew(hn_dim=hn_dim, 
                                         num_atom_types=num_atom_types,
                                         num_bond_types=num_bond_types)
        
        self.reset_parameters()

    def reset_parameters(self):
        self.time_mapping.reset_parameters()
        self.atom_mapping.reset_parameters()
        self.atom_time_mapping.reset_parameters()
        self.gnn.reset_parameters()
        self.score_head.reset_parameters()
        
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
        pos: Tensor,
        edge_index_local: Tensor,
        edge_index_global: Tensor,
        edge_attr_local: OptTensor = None,
        edge_attr_global: OptTensor = None,
        batch: OptTensor = None,
        batch_edge_global: OptTensor = None) -> Dict[Tensor, Tensor]:
        
        
        pos = pos - scatter_mean(pos, index=batch, dim=0)[batch]
        # t: (batch_size,)
        temb = self.time_mapping(t)
        tnode = temb[batch]
        
        if batch is None:
            batch = torch.zeros(x.size(0), device=x.device, dtype=torch.long)
     
        s = self.atom_mapping(x)
        s = self.atom_time_mapping(s + tnode)
                
        # local
        edge_attr_local = self.calculate_edge_attrs(edge_index=edge_index_local, edge_attr=None, pos=pos)        
        # global
        edge_attr_global_transformed = self.calculate_edge_attrs(edge_index=edge_index_global, edge_attr=edge_attr_global, pos=pos)
        
        v = torch.zeros(size=(x.size(0), 3, self.vdim), device=s.device)

        out = self.gnn(
            s=s, v=v, p=pos,
            edge_index_local=edge_index_local, edge_attr_local=edge_attr_local,
            edge_index_global=edge_index_global, edge_attr_global=edge_attr_global_transformed,
            batch=batch
        )
        
        out = self.score_head(x=out, pos=out['p'], batch=batch, edge_index_global=edge_index_global)
        
        out['coords_perturbed'] = pos
        out['atoms_perturbed'] = x
        out['bonds_perturbed'] = edge_attr_global
        
        return out
    
    
    
if __name__ == "__main__":
    pass