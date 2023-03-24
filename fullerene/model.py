import math
from typing import Tuple, Dict, Optional, List

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.nn.inits import reset
from torch_geometric.typing import OptTensor
from torch_scatter import scatter

from e3moldiffusion.modules import DenseLayer, GatedEquivBlock
from e3moldiffusion.molfeat import AtomEncoder, BondEncoder, get_bond_feature_dims
from e3moldiffusion.gnn import EncoderGNN

from fullerene.dataset import Fullerene

BOND_FEATURE_DIMS = get_bond_feature_dims()[0]

class ConservativePotential(nn.Module):
    def __init__(self,
                 hn_dim: Tuple[int, int] = (64, 16),
                 t_dim: int = 64,
                 edge_dim: int = 16,
                 rbf_dim: int = 16,
                 cutoff: float = 10.0,
                 num_layers: int = 5,
                 use_norm: bool = True,
                 use_cross_product: bool = False,
                 use_all_atom_features: bool = True,
                 fully_connected: bool = False,
                 local_global_model: bool = True,
                 vector_aggr: str = "mean",
                 ):
        super(ConservativePotential, self).__init__()

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
        
        self.downstream = GatedEquivBlock(in_dims=hn_dim,
                                          out_dims=(1, None),
                                          use_mlp=True
                                          )
        
    def reset_parameters(self):
        self.atom_encoder.reset_parameters()
        reset(self.t_mapping)
        if self.edge_dim:
            self.edge_encoder.reset_parameters()
        
        self.gnn.reset_parameters()
        self.downstream.reset_parameters()

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
        
        energy, _ = self.downstream(x=(out["s"], out["v"]))
        energy = scatter(src=energy, index=batch,
                         dim=0, dim_size=int(batch.max()) + 1,
                         reduce="add")
        return energy
    

if __name__ == "__main__":
    dataset = Fullerene(root="/Users/tuanle/Desktop/projects/e3moldiffusion/catcher/data/Hexane",
                        dataset="hexane")
    print(dataset)
    from torch_geometric.loader import DataLoader
    from torch_sparse import coalesce
    
    loader = DataLoader(dataset=dataset, batch_size=16)
    data = next(iter(loader))
    model = ConservativePotential(hn_dim=(64, 16),
                                  t_dim=64,
                                  edge_dim=16,
                                  rbf_dim=16,
                                  cutoff=10.0,
                                  num_layers=5,
                                  use_norm=True,
                                  use_all_atom_features=True,
                                  use_cross_product=False,
                                  fully_connected=False,
                                  local_global_model=True
                                  )
    
    t = torch.zeros((32, 64))
    
    def coalesce_edges(edge_index, bond_edge_index, bond_edge_attr, n):
        edge_attr = torch.full(size=(edge_index.size(-1), ), fill_value=BOND_FEATURE_DIMS + 1, device=edge_index.device, dtype=torch.long)
        edge_index = torch.cat([edge_index, bond_edge_index], dim=-1)
        edge_attr =  torch.cat([edge_attr, bond_edge_attr], dim=0)
        edge_index, edge_attr = coalesce(index=edge_index, value=edge_attr, m=n, n=n, op="min")
        return edge_index, edge_attr
    
    edge_index_local, edge_attr_local = data.edge_index, data.edge_attr
    edge_index_global = data.edge_index_fc
    edge_index_global, edge_attr_global = coalesce_edges(edge_index=edge_index_global,
                                                         bond_edge_index=edge_index_local,
                                                         bond_edge_attr=edge_attr_local,
                                                         n=data.num_nodes
                                                         )

    x, pos, batch = data.x, data.pos, data.batch

    pos.requires_grad = True
    energy = model(x=x,
                   t=t,
                   pos=pos,
                   edge_index_local=edge_index_local,
                   edge_index_global=edge_index_global,
                   edge_attr_local=edge_attr_local,
                   edge_attr_global=edge_attr_global,
                   batch=batch)
    
    grad_outputs = [torch.ones_like(energy)]
    scores = torch.autograd.grad(
        outputs=[energy],
        inputs=[pos],
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
    )[0]

    print(pos.shape, scores.shape)
