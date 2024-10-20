from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from e3moldiffusion.modules import DenseLayer
from torch import Tensor


def gaussian_entropy(logvar):
    const = 0.5 * float(logvar.size(1)) * (1.0 + np.log(np.pi * 2))
    ent = 0.5 * logvar.sum(dim=1, keepdim=False) + const
    return ent


def standard_normal_logprob(z):
    log_z = -0.5 * np.log(2 * np.pi)
    out = log_z - z.pow(2) / 2
    return out

def compute_kernel(x, y):
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    x_ = x.unsqueeze(1) # (x_size, 1, dim)
    y_ = y.unsqueeze(0) # (1, y_size, dim)
    tiled_x = x_.expand(x_size, y_size, dim)
    tiled_y = y_.expand(x_size, y_size, dim)
    kernel_input = (tiled_x - tiled_y).pow(2).mean(2)/float(dim)
    kernel = torch.exp(-kernel_input) # (x_size, y_size)
    return kernel

def compute_mmd(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    mmd = x_kernel.mean() + y_kernel.mean() - 2*xy_kernel.mean()
    return mmd

class LatentCache:
    def __init__(self) -> None:
        self.latent_list = []
        self.num_nodes_list = []

    def empty_cache(self):
        self.latent_list = []
        self.num_nodes_list = []

    def update_cache(self, latent, num_nodes):
        self.latent_list.append(latent)
        self.num_nodes_list.append(num_nodes)

    def get_cached_latents(self, n: int):
        ids = np.arange(len(self.latent_list))
        np.random.shuffle(ids)
        select_ids = ids[:n]
        selected_latents = [self.latent_list[i] for i in select_ids]
        selected_num_nodes = torch.tensor([self.num_nodes_list[i] for i in select_ids])
        selected_latents = torch.concat(selected_latents, dim=0)
        assert selected_latents.shape[0] == n
        assert selected_num_nodes.shape[0] == n
        return selected_latents, selected_num_nodes


class LatentMLP(nn.Module):
    """_summary_
    A simple MLP that accepts data input as well as one-dimensional time variable.
    Args:
        nn (_type_): _description_
    """
    def __init__(self, dim: int, num_layers: int) -> None:
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.activation = nn.SiLU()
        self.time_embedding = DenseLayer(1, dim)
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_layers)])
        self.lins = nn.ModuleList([DenseLayer(2 * dim, dim) for _ in range(num_layers)])

    def forward(self, z: Tensor, t: Tensor) -> Tensor:
        temb = self.time_embedding(t)
        for n, m in zip(self.norms, self.lins):
            zin = torch.cat([z, temb], dim=-1)
            zout = m(zin)
            zout = n(zout)
            zout = self.activation(zout)
            z = zout + z
        return z


class LatentNormalizingFlow(nn.Module):
    """_summary_
    Learns a flow from encoded latent z=enc(x) (data) towards a noise distribution w
    Args:
        nn (_type_): _description_
    """

    def __init__(self, flows: List[nn.Module]):
        super().__init__()

        self.flows = nn.ModuleList(flows)

    def f(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Maps observation z to noise variable w.
        Additionally, computes the log determinant
        of the Jacobian for this transformation.
        Invers of g."""
        w, sum_log_abs_det = z, torch.zeros((z.size(0), 1), device=z.device)
        for flow in self.flows:
            w, log_abs_det = flow.f(w)
            sum_log_abs_det += log_abs_det
        return w, sum_log_abs_det

    def g(self, w: torch.Tensor) -> torch.Tensor:
        """Maps latent variable w to encoded observation z=enc(x).
        Inverse of f."""
        with torch.no_grad():
            z = w
            for flow in reversed(self.flows):
                z = flow.g(z)
            return z

    def g_steps(self, w: torch.Tensor) -> List[torch.Tensor]:
        """Maps noise variable w to observation z
        and stores intermediate results."""
        zs = [w]
        for flow in reversed(self.flows):
            zs.append(flow.g(zs[-1]))
        return zs

    def __len__(self) -> int:
        return len(self.flows)


class AffineCouplingLayer(nn.Module):
    def __init__(
        self,
        d,
        intermediate_dim,
        swap,
    ):
        super().__init__()
        self.d = d - (d // 2)
        self.intermediate_dim = intermediate_dim
        self.swap = swap

        self.net_s_t = nn.Sequential(
            DenseLayer(self.d, intermediate_dim),
            nn.LayerNorm(intermediate_dim),
            nn.SiLU(),
            DenseLayer(intermediate_dim, intermediate_dim),
            nn.LayerNorm(intermediate_dim),
            nn.SiLU(),
            DenseLayer(intermediate_dim, (d - self.d) * 2),
        )

    def swap_dims(self, input: torch.Tensor):
        if self.swap:
            input = torch.cat([input[:, self.d :], input[:, : self.d]], 1)
        return input

    def f(self, z: torch.Tensor) -> torch.Tensor:
        """f : z -> w. The inverse of g."""
        z = self.swap_dims(z)
        z2, z1 = z.chunk(2, dim=-1)
        s, t = self.net_s_t(z1).chunk(2, dim=-1)
        w1, w2 = z1, z2 * torch.exp(s) + t
        log_det = s.sum(-1, keepdim=True)
        return torch.cat((w1, w2), dim=-1), log_det

    def g(self, w: torch.Tensor) -> torch.Tensor:
        """g : w -> z. The inverse of f."""
        w = self.swap_dims(w)
        w1, w2 = w.chunk(2, dim=-1)
        s, t = self.net_s_t(w1).chunk(2, dim=-1)
        z1, z2 = w1, (w2 - t) * torch.exp(-s)
        return torch.cat((z2, z1), dim=-1)


def build_latent_flow(
    latent_dim: int, latent_flow_hidden_dim: int, latent_flow_depth: int
):
    flows = []
    for i in range(latent_flow_depth):
        flows.append(
            AffineCouplingLayer(
                d=latent_dim, intermediate_dim=latent_flow_hidden_dim, swap=(i % 2 == 0)
            )
        )
    return LatentNormalizingFlow(flows=flows)


def diffusion_loss(inputdict: Dict, kind: str = "l2"):
    if kind == "l1":
        prior_loss = F.l1_loss(inputdict.get("z_pred"), inputdict.get("z_true"))
    else:
        prior_loss = F.mse_loss(inputdict.get("z_pred"), inputdict.get("z_true"))
    return prior_loss


def flow_loss(inputdict: Dict):
    # P(z), Prior probability, parameterized by the flow: z -> w.
    w, delta_log_pw = inputdict.get("w"), inputdict.get("delta_log_pw")
    batch_size = w.shape[0]
    # compute probability if the flow was able to map the latent to a gaussian prior
    log_pw = (
        standard_normal_logprob(w).view(batch_size, -1).sum(dim=1, keepdim=True)
    )  # (B, 1)
    # compute log p(z) by applying change of variable
    log_pz = log_pw + delta_log_pw.view(batch_size, 1)  # (B, 1)
    prior_loss = -log_pz.mean()
    return prior_loss

def mmd_loss(inputdict: Dict):
    prior_loss = compute_mmd(x=inputdict.get("z_true"), y=torch.randn_like(inputdict.get("z_true")))
    return prior_loss

def vae_loss(inputdict: Dict):
    mu, logvar = inputdict.get("mu"), inputdict.get("logvar")
    prior_loss = torch.mean(
        -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1), dim=0
    )
    return prior_loss

    
class PriorLatentLoss(nn.Module):
    def __init__(self, kind: str) -> None:
        super().__init__()
        assert kind in ["vae", "mmd", "nflow", "diffusion"]
        self.kind = kind
        
    def forward(self, inputdict: Dict) -> Tensor:
        if self.kind == "vae":
            return vae_loss(inputdict)
        elif self.kind == "mmd":
            return mmd_loss(inputdict)
        elif self.kind == "nflow":
            return flow_loss(inputdict)
        elif self.kind == "diffusion":
            return diffusion_loss(inputdict, kind="l1")


def get_latent_model(hparams):
    
    mode = hparams.get("latentmodel")
    
    assert mode in ["vae", "mmd", "nflow", "diffusion"]
    if mode == "diffusion":
        model = LatentMLP(dim=hparams["latent_dim"], num_layers=hparams["latent_layers"])
    elif mode == "nflow":
        model = build_latent_flow(latent_dim=hparams["latent_dim"],
                                  latent_flow_hidden_dim=hparams["latent_dim"],
                                  latent_flow_depth=hparams["latent_layers"]
                                  )
    else:
        model = None
        
    return model