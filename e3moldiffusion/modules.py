import math
from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.init import kaiming_uniform_, zeros_
from torch_geometric.nn.inits import reset
from torch_scatter import scatter_mean


class DenseLayer(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        activation: Union[Callable, nn.Module] = None,
        weight_init: Callable = kaiming_uniform_,
        bias_init: Callable = zeros_,
    ):
        self.weight_init = weight_init
        self.bias_init = bias_init
        super(DenseLayer, self).__init__(in_features, out_features, bias)

        self.activation = activation
        if self.activation is None:
            self.activation = nn.Identity()

    def reset_parameters(self):
        # https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py#L106
        self.weight_init(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            self.bias_init(self.bias)

    def forward(self, input: torch.Tensor):
        y = F.linear(input, self.weight, self.bias)
        y = self.activation(y)
        return y


class GatedEquivBlock(nn.Module):
    def __init__(
        self,
        in_dims: Tuple[int, int],
        out_dims: Tuple[int, Optional[int]],
        hs_dim: Optional[int] = None,
        hv_dim: Optional[int] = None,
        norm_eps: float = 1e-6,
        use_mlp: bool = False
    ):
        super(GatedEquivBlock, self).__init__()
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.vo = 0 if self.vo is None else self.vo

        self.hs_dim = hs_dim or max(self.si, self.so)
        self.hv_dim = hv_dim or max(self.vi, self.vo)
        self.norm_eps = norm_eps

        self.use_mlp = use_mlp

        self.Wv0 = DenseLayer(self.vi, self.hv_dim + self.vo, bias=False)

        if not use_mlp:
            self.Ws = DenseLayer(self.hv_dim + self.si, self.vo + self.so, bias=True)
        else:
            self.Ws = nn.Sequential(
                DenseLayer(self.hv_dim + self.si, self.si, bias=True, activation=nn.SiLU()),
                DenseLayer(self.si, self.vo + self.so, bias=True)
            )
            if self.vo > 0:
                self.Wv1 = DenseLayer(self.vo, self.vo, bias=False)
            else:
                self.Wv1 = None

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.Ws)
        reset(self.Wv0)
        if self.use_mlp:
            if self.vo > 0:
                reset(self.Wv1)


    def forward(self, x: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:

        s, v = x
        vv = self.Wv0(v)

        if self.vo > 0:
            vdot, v = vv.split([self.hv_dim, self.vo], dim=-1)
        else:
            vdot = vv

        vdot = torch.clamp(torch.pow(vdot, 2).sum(dim=1), min=self.norm_eps) #.sqrt()

        s = torch.cat([s, vdot], dim=-1)
        s = self.Ws(s)
        if self.vo > 0:
            gate, s = s.split([self.vo, self.so], dim=-1)
            v = gate.unsqueeze(1) * v
            if self.use_mlp:
                v = self.Wv1(v)

        return s, v


class LayerNorm(nn.Module):
    def __init__(self, dims: Tuple[int, Optional[int]], eps: float = 1e-6, affine: bool = True):
        super().__init__()

        self.dims = dims
        self.sdim, self.vdim = dims
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.Tensor(self.sdim))
            self.bias = nn.Parameter(torch.Tensor(self.sdim))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            self.weight.data.fill_(1.0)
            self.bias.data.fill_(0.0)

    def forward(self, x: Tuple[Tensor, Optional[Tensor]], batch: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        s, v = x
        batch_size = int(batch.max()) + 1
        smean = s.mean(dim=-1, keepdim=True)
        smean = scatter_mean(smean, batch, dim=0, dim_size=batch_size)
    
        s = s - smean[batch]

        var = (s * s).mean(dim=-1, keepdim=True)
        var = scatter_mean(var, batch, dim=0, dim_size=batch_size)
        var = torch.clamp(var, min=self.eps) # .sqrt()
        sout = s / var[batch]

        if self.weight is not None and self.bias is not None:
            sout = sout * self.weight + self.bias

        if v is not None:
            vmean = torch.pow(v, 2).sum(dim=1, keepdim=True).mean(dim=-1, keepdim=True)
            vmean = scatter_mean(vmean, batch, dim=0, dim_size=batch_size)
            vmean = torch.clamp(vmean, min=self.eps)
            vout = v / vmean[batch]
        else:
            vout = None

        out = sout, vout

        return out

    def __repr__(self):
        return (f'{self.__class__.__name__}(dims={self.dims}, '
                f'affine={self.affine})')
        
        
class PolynomialCutoff(nn.Module):
    def __init__(self, cutoff, p: int = 6):
        super(PolynomialCutoff, self).__init__()
        self.cutoff = cutoff
        self.p = p

    @staticmethod
    def polynomial_cutoff(
        r: Tensor,
        rcut: float,
        p: float = 6.0
    ) -> Tensor:
        """
        Polynomial cutoff, as proposed in DimeNet: https://arxiv.org/abs/2003.03123
        """
        if not p >= 2.0:
            print(f"Exponent p={p} has to be >= 2.")
            print("Exiting code.")
            exit()

        rscaled = r / rcut

        out = 1.0
        out = out - (((p + 1.0) * (p + 2.0) / 2.0) * torch.pow(rscaled, p))
        out = out + (p * (p + 2.0) * torch.pow(rscaled, p + 1.0))
        out = out - ((p * (p + 1.0) / 2) * torch.pow(rscaled, p + 2.0))

        return out * (rscaled < 1.0).float()

    def forward(self, r):
        return self.polynomial_cutoff(r=r, rcut=self.cutoff, p=self.p)

    def __repr__(self):
        return f"{self.__class__.__name__}(cutoff={self.cutoff}, p={self.p})"
    

class BesselExpansion(nn.Module):
    def __init__(
        self, max_value: float, K: int = 20
    ):
        super(BesselExpansion, self).__init__()
        self.max_value = max_value
        self.K = K
        frequency = math.pi * torch.arange(start=1, end=K + 1)
        self.register_buffer("frequency", frequency)
        self.reset_parameters()

    def reset_parameters(self):
        self.frequency.data = math.pi * torch.arange(start=1, end=self.K + 1)

    def forward(self, x: Tensor) -> Tensor:
        """
        Bessel RBF, as proposed in DimeNet: https://arxiv.org/abs/2003.03123
        """
        ax = x.unsqueeze(-1) / self.max_value
        ax = ax * self.frequency
        sinax = torch.sin(ax)
        norm = torch.where(
            x == 0.0, torch.tensor(1.0, dtype=x.dtype, device=x.device), x
        )
        out = sinax / norm[..., None]
        out *= math.sqrt(2 / self.max_value)
        return out


class ChebyshevExpansion(nn.Module):
    def __init__(self, max_value: float, embedding_dim: int):
        super(ChebyshevExpansion, self).__init__()
        self.embedding_dim = embedding_dim
        self.max_value = max_value
        self.shift_scale = lambda x: 2 * x / max_value - 1.0

    @staticmethod
    def chebyshev_recursion(x, n):
        if x.ndim == 1:
            x = x.unsqueeze(-1)
        if not n > 2:
            print(f"Naural exponent n={n} has to be > 2.")
            print("Exiting code.")
            exit()

        t_n_1 = torch.ones_like(x)
        t_n = x
        ts = [t_n_1, t_n]
        for _ in range(n - 2):
            t_n_new = 2 * x * t_n - t_n_1
            t_n_1 = t_n
            t_n = t_n_new
            ts.append(t_n_new)
            
        basis_functions = torch.cat(ts, dim=-1)
        return basis_functions
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.shift_scale(x)
        x = self.chebyshev_recursion(x, n=self.embedding_dim)
        return x
    
    @classmethod
    def plot_chebyshev_functions(self, max_value: float = 1.0, embedding_dim: Optional[int] = 20):
        import matplotlib.pyplot as plt
        if embedding_dim is None:
            embedding_dim = self.embedding_dim     
        t = torch.linspace(0, max_value, 1000)
        basis_functions = self(x=t)
        for basis in range(basis_functions.size(1)):
            plt.plot(t, basis_functions[:, basis].cpu().numpy(), label=r"b_{}".format(basis))
        plt.legend()
        plt.show()
        return None
    
    def __repr__(self):
        return f"{self.__class__.__name__}(embedding_dim={self.embedding_dim}, max_value={self.max_value})"