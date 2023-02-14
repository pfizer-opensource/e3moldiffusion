import math
from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn


def get_timestep_embedding(timesteps: Tensor, embedding_dim: int) -> Tensor:
    """
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(0, half_dim) * -emb).to(timesteps.device)
    emb = torch.matmul(1.0 * timesteps.reshape(-1, 1), emb.reshape(1, -1))
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    return emb


def get_ddpm_params(
    beta_min: float = 0.1, beta_max: float = 20.0, num_scales: int = 1000
) -> dict:
    """Get betas and alphas --- parameters used in the original DDPM paper."""
    num_diffusion_timesteps = 1000
    # parameters need to be adapted if number of time steps differs from 1000
    beta_start = beta_min / num_scales
    beta_end = beta_max / num_scales
    betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)

    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)

    sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
    sqrt_1m_alphas_cumprod = np.sqrt(1.0 - alphas_cumprod)

    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "sqrt_alphas_cumprod": sqrt_alphas_cumprod,
        "sqrt_1m_alphas_cumprod": sqrt_1m_alphas_cumprod,
        "beta_min": beta_start * (num_diffusion_timesteps - 1),
        "beta_max": beta_end * (num_diffusion_timesteps - 1),
        "num_diffusion_timesteps": num_diffusion_timesteps,
    }


class VPSDE(nn.Module):
    def __init__(
        self,
        beta_min: float = 0.1,
        beta_max: float = 20.0,
        N: int = 1000,
        scaled_reverse_posterior_sigma: bool = False,
    ):
        """Construct a Variance Preserving SDE.
        Args:
          beta_min: value of beta(0)
          beta_max: value of beta(1)
          N: number of discretization steps
        """
        super().__init__()
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.N = N
        self.scaled_reverse_posterior_sigma = scaled_reverse_posterior_sigma

        discrete_betas = torch.linspace(beta_min / N, beta_max / N, N)
        sqrt_betas = torch.sqrt(discrete_betas)
        alphas = 1.0 - discrete_betas
        sqrt_alphas = torch.sqrt(alphas)
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.nn.functional.pad(
            alphas_cumprod[:-1], (1, 0), value=1.0
        )
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_1m_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

        if scaled_reverse_posterior_sigma:
            rev_variance = (
                discrete_betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
            )
            rev_variance[0] = rev_variance[1] / 2.0
            reverse_posterior_sigma = torch.sqrt(rev_variance)
        else:
            reverse_posterior_sigma = torch.sqrt(discrete_betas)

        self.register_buffer("discrete_betas", discrete_betas)
        self.register_buffer("sqrt_betas", sqrt_betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("sqrt_alphas", sqrt_alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", sqrt_alphas_cumprod)
        self.register_buffer("sqrt_1m_alphas_cumprod", sqrt_1m_alphas_cumprod)
        self.register_buffer("reverse_posterior_sigma", reverse_posterior_sigma)

    @property
    def T(self):
        return 1.0

    def sde(self, x: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        expand_axis = len(x.size()) - 1
        beta_t = self.beta_min + t * (self.beta_max - self.beta_min)

        for _ in range(expand_axis):
            beta_t = beta_t.unsqueeze(-1)

        drift = -0.5 * beta_t * x
        diffusion = torch.sqrt(beta_t)
        return drift, diffusion

    def marginal_prob(self, x: Tensor, t: Tensor):
        expand_axis = len(x.size()) - 1
        log_mean_coeff = (
            -0.25 * t**2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
        )

        for _ in range(expand_axis):
            log_mean_coeff = log_mean_coeff.unsqueeze(-1)

        mean = torch.exp(log_mean_coeff) * x
        std = torch.sqrt(1.0 - torch.exp(2.0 * log_mean_coeff))
        return mean, std

    def prior_sampling(self, shape) -> Tensor:
        return torch.randn(*shape, device=self.alphas.device)

    def prior_logp(self, z: Tensor) -> Tensor:
        shape = z.shape
        N = np.prod(shape[1:])
        logps = (
            -N / 2.0 * np.log(2 * np.pi)
            - (z**2).view(z.size(0), -1).sum(dim=-1) / 2.0
        )
        return logps

    def discretize(self, x: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        """DDPM discretization."""
        expand_axis = len(x.size()) - 1
        timestep = (t * (self.N - 1) / self.T).long()

        sqrt_alpha = self.sqrt_alphas[timestep]
        sqrt_beta = self.sqrt_betas[timestep]

        for _ in range(expand_axis):
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_beta = sqrt_beta.unsqueeze(-1)

        f = sqrt_alpha * x - x
        G = sqrt_beta
        return f, G


class VPAncestralSamplingPredictor:
    """The ancestral sampling predictor. Only supports VP SDE."""

    def __init__(self, sde: VPSDE):
        self.sde = sde

    def update_fn(
        self, x: Tensor, score: Tensor, t: Tensor, noise: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        assert x.shape == score.shape
        expand_axis = len(x.size()) - 1

        sde = self.sde
        timestep = (t * (sde.N - 1) / sde.T).long()
        sqrt_alpha = sde.sqrt_alphas[timestep]
        beta = sde.discrete_betas[timestep]
        sqrt_1m_alphas_cumprod = sde.sqrt_1m_alphas_cumprod[timestep]
        reverse_posterior_sigma = sde.reverse_posterior_sigma[timestep]

        for _ in range(expand_axis):
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_1m_alphas_cumprod = sqrt_1m_alphas_cumprod.unsqueeze(-1)
            beta = beta.unsqueeze(-1)
            reverse_posterior_sigma = reverse_posterior_sigma.unsqueeze(-1)

        scaling = beta / sqrt_1m_alphas_cumprod
        x_mean = (x - scaling * score) / sqrt_alpha

        if noise is None:
            noise = torch.randn_like(x)

        x = x_mean + reverse_posterior_sigma * noise

        return x, x_mean

# ToDo: More samplers, i.e. Predictors and Correctors.

if __name__ == "__main__":
    from torch_scatter import scatter_mean

    sde = VPSDE(beta_min=0.1, beta_max=20.0, N=1000)

    x = torch.randn(5, 2)
    eps_min = 1e-3

    t = torch.rand(x.size(0))
    t = t * (1.0 - eps_min) + eps_min

    print(x)
    print("---" * 10)
    meant, sigmat = sde.marginal_prob(x=x, t=t)
    print(t)
    print()
    print(meant)
    print(sigmat)

    drift, diffusion = sde.sde(x=x, t=t)
    print()
    print(drift)
    print(diffusion)

    # in case we have a batch of 16 point clouds
    bs = 16

    batch = torch.sort(torch.randint(low=0, high=bs, size=(bs * 30,)))[0]
    num_graphs = torch.bincount(batch)
    x = torch.randn(size=(len(batch), 3), dtype=torch.float32)

    # translate to 0-COM
    x = x - scatter_mean(x, batch, dim=0)[batch]
    print(scatter_mean(x, batch, dim=0).norm())

    noise = torch.randn_like(x)
    noise = noise - scatter_mean(noise, batch, dim=0)[batch]

    t = torch.rand(bs)
    t = t * (sde.T - eps_min) + eps_min
    t = t[batch]

    meant, sigmat = sde.marginal_prob(x=x, t=t)

    # sampling xt | x0
    xt = meant + sigmat * noise
    