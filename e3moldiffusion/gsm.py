import math
from typing import Tuple, Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.nn.inits import reset
from torch_geometric.typing import OptTensor
from torch_scatter import scatter

from e3moldiffusion.convs import EQGATRBFConv
from e3moldiffusion.modules import DenseLayer, GatedEquivBlock
from e3moldiffusion.molfeat import AtomEncoder, BondEncoder
from e3moldiffusion.gnn import EncoderGNN
from e3moldiffusion.potential import SimplePotential

import functorch
# https://pytorch.org/functorch/0.1.1/


""" Model Script for Generalized Score Matching on Molecular Systems """

def keep_grad(output, input, grad_outputs=None):
    return torch.autograd.grad(output, input, grad_outputs=grad_outputs,
                               retain_graph=True, create_graph=True)[0]

def exact_jacobian_trace(fx, x):
    vals = []
    # [N, 3]
    for i in range(x.size(1)):
        fxi = fx[:, i]
        dfxi_dxi = keep_grad(fxi, x, torch.ones_like(fxi))[:, i][:, None]
        vals.append(dfxi_dxi)
    vals = torch.cat(vals, dim=1)
    vals = vals.sum(dim=1)
    return vals


class NonConservativeScoreNetwork(nn.Module):
    def __init__(self,
                 hn_dim: Tuple[int, int] = (64, 16),
                 edge_dim: int = 16,
                 rbf_dim: int = 16,
                 cutoff: float = 10.0,
                 num_layers: int = 5,
                 use_norm: bool = True,
                 use_cross_product: bool = False,
                 use_all_atom_features: bool = True,
                 fully_connected: bool = False,
                 local_global_model: bool = False,
                 vector_aggr: str = "mean",
                 dist_score: bool = False
                 ):
        super(NonConservativeScoreNetwork, self).__init__()

        self.atom_encoder = AtomEncoder(emb_dim=hn_dim[0],
                                        use_all_atom_features=use_all_atom_features,
                                        max_norm=None)

        if edge_dim is None or 0:
            self.edge_dim = 0
        else:
            self.edge_dim = edge_dim

        if self.edge_dim:
            self.edge_encoder = BondEncoder(emb_dim=edge_dim, max_norm=None)
        else:
            self.edge_encoder = None
        
        self.sdim, self.vdim = hn_dim

        self.local_global_model = local_global_model
        self.fully_connected = fully_connected
        
        self.gnn = EncoderGNN(
            hn_dim=hn_dim,
            cutoff=cutoff,
            rbf_dim=rbf_dim,
            edge_dim=edge_dim,
            num_layers=num_layers,
            use_norm=use_norm,
            use_cross_product=use_cross_product,
            vector_aggr=vector_aggr,
            fully_connected=fully_connected,
            local_global_model=local_global_model
        )
        
        self.coords_score = DenseLayer(in_features=hn_dim[1], out_features=1)

        if dist_score:
            self.dist_score = DenseLayer(in_features=hn_dim[0], out_features=1)
        else:
            self.dist_score = None
            
        self.conservative = True
        
        self.reset_parameters()

    def reset_parameters(self):
        self.atom_encoder.reset_parameters()
        if self.edge_dim:
            self.edge_encoder.reset_parameters()        
        self.gnn.reset_parameters()
        self.coords_score.reset_parameters()
        if self.dist_score is not None:
            self.dist_score.reset_parameters()

    def calculate_edge_attrs(self, edge_index: Tensor, edge_attr: OptTensor, pos: Tensor):
        source, target = edge_index
        r = pos[target] - pos[source]
        d = torch.clamp(torch.pow(r, 2).sum(-1), min=1e-6).sqrt()
        r_norm = torch.div(r, d.unsqueeze(-1))
        edge_attr = (d, r_norm, edge_attr)
        return edge_attr
    
    def forward(self,
                x: Tensor,
                pos: Tensor,
                edge_index_local: Tensor,
                edge_index_global: Tensor,
                edge_attr_local: OptTensor = None,
                edge_attr_global: OptTensor = None,
                batch: OptTensor = None):

        if batch is None:
            batch = torch.zeros(x.size(0), device=x.device, dtype=torch.long)
     
        if self.edge_dim > 0:
            edge_attr_local = self.edge_encoder(edge_attr_local)
            # edge_attr_global = self.edge_encoder(edge_attr_global)
            edge_attr_global = None
         
        # local
        edge_attr_local = self.calculate_edge_attrs(edge_index=edge_index_local, edge_attr=edge_attr_local, pos=pos)        
        # global
        # edge_attr_global = self.calculate_edge_attrs(edge_index=edge_index_global, edge_attr=edge_attr_global, pos=pos)

        s = self.atom_encoder(x)
        v = torch.zeros(size=(x.size(0), 3, self.vdim), device=s.device)

        out = self.gnn(
            s=s, v=v,
            edge_index_local=edge_index_local, edge_attr_local=edge_attr_local,
            edge_index_global=edge_index_global, edge_attr_global=edge_attr_global,
            batch=batch
        )
        
        if not self.conservative:
            score = self.coords_score(out["v"]).squeeze()        
            if self.dist_score is not None:
                source, target = edge_index_global
                r = pos[source] - pos[target]
                sd = F.silu(out["s"])
                sd = self.dist_score(sd)
                sd = sd[source] + sd[target]
                sr = sd * r
                sr = scatter(src=sr, index=target, dim=0, dim_size=s.size(0), reduce="add").squeeze()
                score = score + sr 
        else:
            energy = out["s"]
            energy = self.dist_score(energy)
            energy = scatter(energy, batch, dim=0, dim_size=int(batch.max() + 1), reduce="add")
            # score = keep_grad(output=energy, input=pos, grad_outputs=torch.ones_like(energy))
                
        return energy
    
    
class SimplePotentialScoreNetwork(nn.Module):
    def __init__(self,
                 hn_dim: Tuple[int, int] = (64, 16),
                 edge_dim: int = 16,
                 use_all_atom_features: bool = True,
                 **kwargs,
                 ):
        super(SimplePotentialScoreNetwork, self).__init__()

        self.atom_encoder = AtomEncoder(emb_dim=hn_dim[0],
                                        use_all_atom_features=use_all_atom_features,
                                        max_norm=None)

        if edge_dim is None or 0:
            self.edge_dim = 0
        else:
            self.edge_dim = edge_dim

        if self.edge_dim:
            self.edge_encoder = BondEncoder(emb_dim=edge_dim, max_norm=None)
        else:
            self.edge_encoder = None
        
        self.sdim, self.vdim = hn_dim
        self.potential = SimplePotential(
            node_dim=hn_dim[0], edge_dim=edge_dim, vdim=hn_dim[1]
        )
        self.reset_parameters()

    def reset_parameters(self):
        pass  
        
    def forward(self,
                x: Tensor,
                pos: Tensor,
                edge_index_local: Tensor,
                edge_index_global: Tensor,
                edge_attr_local: OptTensor = None,
                edge_attr_global: OptTensor = None,
                batch: OptTensor = None):

        if batch is None:
            batch = torch.zeros(x.size(0), device=x.device, dtype=torch.long)
     
        if self.edge_dim > 0:
            edge_attr_local = self.edge_encoder(edge_attr_local)
            # edge_attr_global = self.edge_encoder(edge_attr_global)
            edge_attr_global = None
         
        s = self.atom_encoder(x)
        
        energy = self.potential(
            x=s, pos=pos,
            edge_index_local=edge_index_local,
            edge_attr_local=edge_attr_local,
            edge_index_global=edge_index_global,
            edge_attr_global=edge_attr_global,
            batch=batch,
        )
        
        # score = keep_grad(output=energy, input=pos, grad_outputs=torch.ones_like(energy))
        
        return energy
    

if __name__ == "__main__":
    x = torch.randn(10, 3, requires_grad=True)
    net = nn.Sequential(DenseLayer(3, 3, activation=nn.SiLU()), DenseLayer(3, 3))
    score = net(x)
    tr = exact_jacobian_trace(fx=score, x=x)
    snorm = score.pow(2).sum(-1).mean()
    jctr = tr.mean()
    print(f"snorm={snorm} , jctr={jctr}")
    
    
    