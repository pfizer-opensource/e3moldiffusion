import math
from typing import Optional, Tuple, Union
import matplotlib.pyplot as plt

import numpy as np
import torch
from torch import Tensor, nn


def get_timestep_embedding(timesteps: torch.Tensor, embedding_dim: int):
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
        scaled_reverse_posterior_sigma: bool = True,
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
        """_summary_
        Eq. 34 in https://arxiv.org/pdf/2011.13456.pdf
        Args:
            x (Tensor): _description_ Continuous data feature tensor
            t (Tensor): _description_ Continuous time variable between 0 and 1
        Returns:
            _type_: _description_
        """
        expand_axis = len(x.size()) - 1
        log_mean_coeff = (
            -0.25 * t**2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
        )

        for _ in range(expand_axis):
            log_mean_coeff = log_mean_coeff.unsqueeze(-1)

        mean = torch.exp(log_mean_coeff) * x
        std = torch.sqrt(1.0 - torch.exp(2.0 * log_mean_coeff))
        return mean, std
    
    @classmethod
    def plot_signal_to_noise(self, beta_min: Optional[float] = 0.1, beta_max: Optional[float] = 20.0):
        import matplotlib.pyplot as plt
        if beta_min is None:
            beta_min = self.beta_min
        if beta_max is None:
            beta_max = self.beta_max 
            
        t = torch.linspace(0, 1, 1000)
        
        log_mean_coeff = (
            -0.25 * t**2 * (beta_max - beta_min) - 0.5 * t * beta_min
        )
        
        signal = torch.exp(log_mean_coeff)
        noise = torch.sqrt(1.0 - torch.exp(2.0 * log_mean_coeff))
        
        plt.title(f"Signal and Noise Schedule for beta_min={beta_min:.2f} and beta_max={beta_max:.2f}")
        plt.plot(t, signal, label="signal")
        plt.plot(t, noise, label="noise")
        plt.legend()
        plt.show()
        

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


def clip_noise_schedule(alphas2, clip_value=0.001):
    """
    For a noise schedule given by alpha^2, this clips alpha_t / alpha_t-1. This may help improve stability during
    sampling.
    """
    alphas2 = np.concatenate([np.ones(1), alphas2], axis=0)

    alphas_step = (alphas2[1:] / alphas2[:-1])

    alphas_step = np.clip(alphas_step, a_min=clip_value, a_max=1.)
    alphas2 = np.cumprod(alphas_step, axis=0)

    return alphas2


def get_beta_schedule(beta_start: float = 1e-4,
                      beta_end: float = 2e-2,
                      num_diffusion_timesteps: int = 1000,
                      kind: str = 'linear',
                      plot: bool = False, 
                      **kwargs):

    if kind == 'quad':
        betas = (
            torch.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=torch.get_default_dtype(),
            )
            ** 2
        )
    elif kind == 'sigmoid':
        betas = torch.linspace(-6, 6, num_diffusion_timesteps)
        betas = torch.sigmoid(betas) * (beta_end - beta_start) + beta_start
    elif kind == 'linear':
        betas = torch.linspace(beta_start, beta_end, num_diffusion_timesteps)
    elif kind == "cosine":
        s = kwargs.get("s")
        if s is None:
            s = 0.008
        steps = num_diffusion_timesteps + 2
        x = torch.linspace(0, num_diffusion_timesteps, steps)
        alphas_cumprod = torch.cos(((x / num_diffusion_timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clip(betas, 0.0, 0.999)
    elif kind == "polynomial":
        s = kwargs.get("s")
        p = kwargs.get("p")
        if s is None:
            s = 1e-4
        if p is None:
            p = 3.
        steps = num_diffusion_timesteps + 1
        x = np.linspace(0, steps, steps)
        alphas2 = (1 - np.power(x / steps, p))**2
        alphas2 = clip_noise_schedule(alphas2, clip_value=0.001)
        precision = 1 - 2 * s
        alphas2 = precision * alphas2 + s
        alphas = np.sqrt(alphas2)
        betas = 1.0 - alphas
        betas = torch.from_numpy(betas).float()

    if plot:
        plt.plot(range(len(betas)), betas)
        plt.xlabel("t")
        plt.ylabel("beta")
        plt.show()

        alphas = 1.0 - betas
        signal_coeff = alphas.cumprod(0)
        noise_coeff = torch.sqrt(1.0 - signal_coeff)
        plt.plot(np.arange(num_diffusion_timesteps), signal_coeff, label="signal")
        plt.plot(np.arange(num_diffusion_timesteps), noise_coeff, label="noise")
        plt.legend()
        plt.show()

    return betas


class DiscreteDDPM(nn.Module):
    def __init__(
        self,
        beta_min: float = 1e-4,
        beta_max: float = 2e-2,
        N: int = 300,
        scaled_reverse_posterior_sigma: bool = True,
        schedule: str = "cosine",
        **kwargs
    ):
        """Constructs discrete Diffusion schedule according to DDPM in Ho et al. (2020).
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
        
        assert schedule in ["linear", "quad", "cosine", "sigmoid", "polynomial"]
        
        discrete_betas = get_beta_schedule(beta_start=beta_min, beta_end=beta_max,
                                           num_diffusion_timesteps=N,
                                           kind=schedule, plot=False)
        
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

    def marginal_prob(self, x: Tensor, t: Tensor):
        """_summary_
        Eq. 4 in https://arxiv.org/abs/2006.11239
        Args:
            x (Tensor): _description_ Continuous data feature tensor
            t (Tensor): _description_ Discrete time variable between 1 and T
        Returns:
            _type_: _description_
        """
        expand_axis = len(x.size()) - 1
        
        assert str(t.dtype) == "torch.int64"
        signal = self.sqrt_alphas_cumprod[t]
        std = self.sqrt_1m_alphas_cumprod[t]
        
        for _ in range(expand_axis):
            signal = signal.unsqueeze(-1)
            std = std.unsqueeze(-1)

        mean = signal * x
        
        return mean, std
    
    def plot_signal_to_noise(self):
        
        beta_min = self.beta_min
        beta_max = self.beta_max 
            
        t = torch.arange(0, self.N)
        signal = self.sqrt_alphas_cumprod[t]
        std = self.sqrt_1m_alphas_cumprod[t]
        
        plt.title(f"Signal and Noise Schedule for beta_min={beta_min} and beta_max={beta_max}")
        plt.plot(t, signal, label="signal")
        plt.plot(t, std, label="noise")
        plt.xlabel("timesteps")
        plt.legend()
        plt.show()
        

class VPAncestralSamplingPredictor:
    """The ancestral sampling predictor. Only supports VP SDE."""

    def __init__(self, sde: Union[VPSDE, DiscreteDDPM]):
        self.sde = sde

    def update_fn(
        self, x: Tensor, score: Tensor, t: Tensor, noise: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """_summary_
        See Algorithm 2 in https://arxiv.org/pdf/2006.11239.pdf
        Args:
            x (Tensor): _description_
            score (Tensor): _description_
            t (Tensor): _description_
            noise (Optional[Tensor], optional): _description_. Defaults to None.

        Returns:
            Tuple[Tensor, Tensor]: _description_
        """
        assert x.shape == score.shape
        expand_axis = len(x.size()) - 1

        sde = self.sde
        
        if "float" in str(t.dtype):
            # discretize continuous time variable
            timestep = (t * (sde.N - 1) / sde.T).long()
        else:
            # in case t is already discrete
            timestep = t
              
        sqrt_alpha = sde.sqrt_alphas[timestep]
        beta = sde.discrete_betas[timestep]
        sqrt_1m_alphas_cumprod = sde.sqrt_1m_alphas_cumprod[timestep]
        reverse_posterior_sigma = sde.reverse_posterior_sigma[timestep]

        for _ in range(expand_axis):
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_1m_alphas_cumprod = sqrt_1m_alphas_cumprod.unsqueeze(-1)
            beta = beta.unsqueeze(-1)
            reverse_posterior_sigma = reverse_posterior_sigma.unsqueeze(-1)

        if noise is None:
            noise = torch.randn_like(x) 
        # Line 4 in Algorithm 2 in https://arxiv.org/pdf/2006.11239.pdf
        scaling = beta / sqrt_1m_alphas_cumprod
        x_mean = (x - scaling * score) / sqrt_alpha
        x = x_mean + reverse_posterior_sigma * noise

        return x, x_mean

# ToDo: More samplers, i.e. Predictors and Correctors.

if __name__ == "__main__":
    from torch_scatter import scatter_mean

    T = 1000
    schedule = "polynomial"
    
    sde = DiscreteDDPM(beta_min=1e-4,
                       beta_max=2e-2,
                       N=T,
                       scaled_reverse_posterior_sigma=True, 
                       schedule=schedule)
    
    sde.plot_signal_to_noise()
    
    plt.plot(range(len(sde.discrete_betas)), sde.discrete_betas, label="betas")
    plt.plot(range(len(sde.alphas)), sde.alphas, label="alphas")
    plt.xlabel("t")
    plt.legend()
    plt.show()
    
    signal = sde.sqrt_alphas_cumprod
    noise = sde.sqrt_1m_alphas_cumprod
    
    D = 5
    indexes = len(signal) // D
    truncated_timesteps = np.array([i * indexes for i in range(0, D + 1)])
    signal_indexes = signal[truncated_timesteps - 1]
    noise_indexes = noise[truncated_timesteps - 1]

    plt.plot(range(len(signal)), signal, label="signal")
    plt.plot(range(len(noise)), noise, label="noise")
    plt.scatter(np.array(truncated_timesteps), signal_indexes)
    plt.scatter(np.array(truncated_timesteps), noise_indexes)
    plt.legend()
    plt.show()
    
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
    print(scatter_mean(noise, batch, dim=0).norm())

    t = torch.randint(low=0, high=T,  size=(bs, ) )
    t = t[batch]

    meant, sigmat = sde.marginal_prob(x=x, t=t)

    # sampling xt | x0
    xt = meant + sigmat * noise
    
    
    # for reverse, check coefficients
    # update rule is as follows:
    # x_{t-1} = 1 / sqrt(alpha_t) * ( x_t - score_scaling(t) * score(x_t, t) ) + \sigma_t * z
    # where z ~ N(0, 1)
    
    score_scaling = sde.discrete_betas / sde.sqrt_1m_alphas_cumprod
    sigma = sde.reverse_posterior_sigma
    a = 1 / sde.sqrt_alphas
     
    plt.plot(range(len(score_scaling)), score_scaling, label="score_scaling")
    plt.plot(range(len(sigma)), sigma, label="sigma")
    plt.plot(range(len(a)), a, label="1/sqrt(alphas)")
    plt.legend()
    plt.show()
