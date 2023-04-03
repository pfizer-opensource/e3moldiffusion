import torch
from torch import nn, Tensor

from typing import Optional, Tuple, Dict
import torch.nn.functional as F 
from torch_geometric.nn.inits import reset
from torch_geometric.typing import OptTensor
from torch_scatter import scatter


from e3moldiffusion.molfeat import AtomEncoder, BondEncoder
from e3moldiffusion.modules import DenseLayer
from e3moldiffusion.gnn import EncoderGNN

# Score Model that is trained to learn 3D coordinates only

class ScoreHead(nn.Module):
    def __init__(self,
                 hn_dim: Tuple[int, int] = (64, 16),
                 conservative: bool = False) -> None:
        super(ScoreHead, self).__init__()
        self.sdim, self.vdim = hn_dim
        self.conservative = conservative
        self.dscore = DenseLayer(self.sdim, 1, bias=False)
        if conservative:
            self.outhead = DenseLayer(self.sdim, 1, bias=False)
        else:
            self.outhead = DenseLayer(self.vdim, 1, bias=False)
            
        self.reset_parameters()
        
    def reset_parameters(self):
        self.dscore.reset_parameters()
        self.outhead.reset_parameters()
        
    def forward(self,
                x: Dict[Tensor, Tensor],
                pos: Tensor,
                batch: Tensor,
                edge_index_global: Tensor) -> Tensor:
        
        bs = int(batch.max()) + 1
        s, v = x["s"], x["v"]
        s = F.silu(s)
        d = self.dscore(s)
        source, target = edge_index_global
        d = d[source] + d[target]
        if self.conservative:
            sr = d
        else:
            r = pos[source] - pos[target]
            sr = d * r
            
        sr = scatter(src=sr, index=target, dim=0, dim_size=s.size(0), reduce="add").squeeze()
        
        if self.conservative:
            energy = self.outhead(s).squeeze()
            energy = sr + energy
            energy = scatter(src=energy, index=batch, dim=0, dim_size=bs, reduce="add")
            score = torch.autograd.grad(energy, pos, create_graph=True, grad_outputs=torch.ones_like(energy))
        else:
            score = self.outhead(v).squeeze() + sr
        
        return score
        
        
class ScoreModel(nn.Module):
    def __init__(self,
                 hn_dim: Tuple[int, int] = (64, 16),
                 t_dim: int = 64,
                 edge_dim: int = 16,
                 rbf_dim: int = 16,
                 cutoff_local: float = 3.0,
                 cutoff_global: float = 10.0,
                 num_layers: int = 5,
                 use_norm: bool = True,
                 use_cross_product: bool = False,
                 use_all_atom_features: bool = True,
                 fully_connected: bool = False,
                 local_global_model: bool = True,
                 vector_aggr: str = "mean"
                 ):
        super(ScoreModel, self).__init__()

        self.atom_encoder = AtomEncoder(emb_dim=hn_dim[0],
                                        use_all_atom_features=use_all_atom_features,
                                        max_norm=1.0)

        if edge_dim is None or 0:
            self.edge_dim = 0
        else:
            self.edge_dim = edge_dim

        if self.edge_dim:
            self.edge_encoder = BondEncoder(emb_dim=edge_dim, max_norm=1.0)
        else:
            self.edge_encoder = None

        self.t_dim = t_dim
        self.t_mapping = nn.Sequential(
            DenseLayer(t_dim, t_dim, activation=nn.SiLU()),
            DenseLayer(t_dim, hn_dim[0], activation=nn.SiLU())
            )
        
        self.sdim, self.vdim = hn_dim

        self.local_global_model = local_global_model
        self.fully_connected = fully_connected
        
        self.gnn = EncoderGNN(
            hn_dim=hn_dim,
            cutoff_local=cutoff_local,
            cutoff_global=cutoff_global,
            rbf_dim=rbf_dim,
            edge_dim=edge_dim,
            num_layers=num_layers,
            use_norm=use_norm,
            use_cross_product=use_cross_product,
            vector_aggr=vector_aggr,
            fully_connected=fully_connected,
            local_global_model=local_global_model
        )
        
        self.score_head = ScoreHead(hn_dim=hn_dim, conservative=False)
            
        self.reset_parameters()

    def reset_parameters(self):
        self.atom_encoder.reset_parameters()
        reset(self.t_mapping)
        if self.edge_dim:
            self.edge_encoder.reset_parameters()
        self.gnn.reset_parameters()
        self.score_head.reset_parameters()
        
    def calculate_edge_attrs(self, edge_index: Tensor, edge_attr: OptTensor, pos: Tensor):
        source, target = edge_index
        r = pos[target] - pos[source]
        d = torch.clamp(torch.pow(r, 2).sum(-1), min=1e-6).sqrt()
        r_norm = torch.div(r, d.unsqueeze(-1))
        edge_attr = (d, r_norm, edge_attr)
        return edge_attr
    
    def forward(self,
                x: Tensor,
                t: Tensor,
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
            edge_attr_global = self.edge_encoder(edge_attr_global)
         
        # local
        edge_attr_local = self.calculate_edge_attrs(edge_index=edge_index_local, edge_attr=edge_attr_local, pos=pos)        
        # global
        edge_attr_global = self.calculate_edge_attrs(edge_index=edge_index_global, edge_attr=edge_attr_global, pos=pos)

        s = self.atom_encoder(x)
        temb = self.t_mapping(t)[batch]
        s = s + temb
        v = torch.zeros(size=(x.size(0), 3, self.vdim), device=s.device)

        out = self.gnn(
            s=s, v=v,
            edge_index_local=edge_index_local, edge_attr_local=edge_attr_local,
            edge_index_global=edge_index_global, edge_attr_global=edge_attr_global,
            batch=batch
        )
        
        score = self.score_head(x=out, pos=pos, batch=batch, edge_index_global=edge_index_global)
             
        return score
    