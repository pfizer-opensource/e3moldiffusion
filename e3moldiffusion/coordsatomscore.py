import math
from typing import Tuple, Dict

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.nn.inits import reset
from torch_geometric.typing import OptTensor
from torch_scatter import scatter

from e3moldiffusion.modules import GatedEquivBlock, DenseLayer
from e3moldiffusion.molfeat import get_atom_feature_dims
from e3moldiffusion.gnn import EncoderGNN


MAX_NUM_ATOM_FEATURES = get_atom_feature_dims()[0] + 1


class AtomEmbedding(nn.Module):
    def __init__(self, out_dim: int) -> None:
        super().__init__()
        
        self.out_dim = out_dim
        embedding = torch.randn(MAX_NUM_ATOM_FEATURES, out_dim)
        embedding /= embedding.norm(p=2, dim=-1, keepdim=True)
        self.embedding = nn.parameter.Parameter(embedding)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        self.embedding.data = torch.randn(MAX_NUM_ATOM_FEATURES, self.out_dim)
        self.embedding.data /= self.embedding.data.norm(p=2, dim=-1, keepdim=True)
        
    def forward(self, x: Tensor, norm_gradient: bool = True) -> Tensor:
        out = self.embedding[x]
        out_norm = out.norm(p=2, dim=-1, keepdim=True)
        if not norm_gradient:
            out_norm = out_norm.detach()
        out /= out_norm
        return out
        
        
class CoordsAtomScoreModel(nn.Module):
    def __init__(self,
                 hn_dim: Tuple[int, int] = (64, 16),
                 t_dim: int = 64,
                 num_layers: int = 5,
                 use_norm: bool = True,
                 use_cross_product: bool = False,
                 vector_aggr: str = "mean",
                 **kwargs) -> None:
        super().__init__()
        
        
        self.atom_embedder = AtomEmbedding(out_dim=hn_dim[0])
        self.t_dim = t_dim
        self.t_mapping = nn.Sequential(
            DenseLayer(t_dim, t_dim, activation=nn.SiLU()),
            DenseLayer(t_dim, hn_dim[0], activation=nn.SiLU())
            )
        self.sdim, self.vdim = hn_dim
        self.gnn = EncoderGNN(hn_dim=hn_dim,
                              edge_dim=0,
                              num_layers=num_layers,
                              use_norm=use_norm,
                              use_cross_product=use_cross_product,
                              vector_aggr=vector_aggr
                              )
        self.out_lin = GatedEquivBlock(in_dims=hn_dim,
                                       out_dims=(MAX_NUM_ATOM_FEATURES + 1, 1),
                                       use_mlp=True
                                       )
        
        
        
        
        
        
        
        