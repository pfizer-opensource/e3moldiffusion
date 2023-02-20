import math
from typing import Tuple

import torch
from torch import Tensor, nn
from torch_geometric.nn.inits import reset
from torch_geometric.typing import OptTensor
from torch_scatter import scatter

from e3moldiffusion.convs import EQGATConv
from e3moldiffusion.modules import DenseLayer, GatedEquivBlock, LayerNorm
from e3moldiffusion.molfeat import AtomEncoder, BondEncoder


class EQGATEncoder(nn.Module):
    def __init__(self,
                 hn_dim: Tuple[int, int] = (64, 16),
                 t_dim: int = 16,
                 edge_dim: int = 0,
                 num_layers: int = 5,
                 energy_preserving: bool = False,
                 use_norm: bool = False,
                 use_cross_product: bool = False,
                 use_mlp_update: bool = False,
                 use_all_atom_features: bool = False
                 ):
        super(EQGATEncoder, self).__init__()

        self.atom_encoder = AtomEncoder(emb_dim=hn_dim[0],
                                        use_all_atom_features=use_all_atom_features)

        if edge_dim is not None:
            self.edge_dim = edge_dim
        else:
            self.edge_dim = 0

        if self.edge_dim:
            self.edge_encoder = BondEncoder(emb_dim=edge_dim)
        else:
            self.edge_encoder = None

        self.t_dim = t_dim
        self.t_mapping = DenseLayer(t_dim, t_dim)
        self.feat_time_merging = DenseLayer(hn_dim[0] + t_dim, hn_dim[0], bias=False)
        
        self.sdim, self.vdim = hn_dim

        self.convs = nn.ModuleList([
            EQGATConv(in_dims=hn_dim,
                      out_dims=hn_dim,
                      edge_dim=self.edge_dim,
                      has_v_in=i>0,
                      use_mlp_update=use_mlp_update,
                      vector_aggr="mean",
                      use_cross_product=use_cross_product
                      )
            for i in range(num_layers)
        ])

        self.use_norm = use_norm
        self.norms = nn.ModuleList([
            LayerNorm(dims=hn_dim) if use_norm else nn.Identity()
            for _ in range(num_layers)
        ])
        self.energy_preserving = energy_preserving

        if energy_preserving:
            self.downstream = GatedEquivBlock(
                in_dims=hn_dim,
                out_dims=(1, None),
                use_mlp=True,
                hs_dim=hn_dim[0],
                hv_dim=hn_dim[1],
            )
        else:
            self.downstream = DenseLayer(in_features=hn_dim[1], out_features=1)

        self.reset_parameters()

    def reset_parameters(self):
        self.atom_encoder.reset_parameters()
        reset(self.t_mapping)
        if self.edge_dim:
            self.edge_encoder.reset_parameters()
        for conv, norm in zip(self.convs, self.norms):
            conv.reset_parameters()
            if self.use_norm:
                norm.reset_parameters()
        self.downstream.reset_parameters()

    def forward(self,
                x: Tensor,
                t: Tensor,
                pos: Tensor,
                edge_index: Tensor,
                edge_attr: OptTensor = None,
                batch: OptTensor = None):

        if batch is None:
            batch = torch.zeros(x.size(0), device=x.device, dtype=torch.long)
        batch_size = int(batch.max()) + 1

        if edge_attr is not None:
            edge_attr = self.edge_encoder(edge_attr)

        source, target = edge_index
        r = pos[target] - pos[source]
        d = torch.clamp(torch.pow(r, 2).sum(-1), min=1e-6).sqrt()
        r = torch.div(r, d.unsqueeze(-1))
        edge_attr_c = (d, r, edge_attr)

        s = self.atom_encoder(x)
        temb = self.t_mapping(t)[batch]
        s = self.feat_time_merging(torch.concat([s, temb], dim=-1))
        v = torch.zeros(size=(x.size(0), 3, self.vdim), device=s.device)

        for i in range(len(self.convs)):
            if self.use_norm:
                s, v = self.norms[i](x=(s, v), batch=batch)
            s, v = self.convs[i](x=(s, v), edge_index=edge_index, edge_attr=edge_attr_c)

        if self.energy_preserving:
            out, _ = self.downstream(x=(s, v))
            out = scatter(src=out, index=batch, dim=0, dim_size=batch_size, reduce="add").squeeze()
        else:
            out = self.downstream(v).squeeze()

        return out
    
    
if __name__ == '__main__':
    
    from e3moldiffusion.molfeat import smiles_or_mol_to_graph
    from e3moldiffusion.sde import get_timestep_embedding, ChebyshevExpansion
    smol = "O1C=C[C@H]([C@H]1O2)c3c2cc(OC)c4c3OC(=O)C5=C4CCC(=O)5"
    data = smiles_or_mol_to_graph(smol)
    print(data)
    x = data.x
    edge_attr = None
    # fully-connected
    n = x.size(0)
    row = torch.arange(n, dtype=torch.long)
    col = torch.arange(n, dtype=torch.long)
    row = row.view(-1, 1).repeat(1, n).view(-1)
    col = col.repeat(n)
    edge_index = torch.stack([row, col], dim=0)
    mask = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, mask]
    
    pos = torch.randn(data.x.size(0), 3)
    
    t = torch.rand(1, )
    timeembeddder = ChebyshevExpansion(max_value=1.0, embedding_dim=16)
    temb = timeembeddder(t)
    
    model = EQGATEncoder(hn_dim=(64, 16),
                         t_dim=16,
                         edge_dim=0,
                         num_layers=4,
                         energy_preserving=False, 
                         use_norm=False)
    
    out = model(x=x,
                pos=pos,
                t=temb,
                edge_index=edge_index,
                edge_attr=edge_attr)
    print(out)