import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from experiments.utils import zero_mean


def sigmoid(x):
    return 1 / (np.exp(-x) + 1)


def clip_noise_schedule(alphas2, clip_value=0.001):
    """
    For a noise schedule given by alpha^2, this clips alpha_t / alpha_t-1. This may help improve stability during
    sampling.
    """
    alphas2 = np.concatenate([np.ones(1), alphas2], axis=0)

    alphas_step = alphas2[1:] / alphas2[:-1]

    alphas_step = np.clip(alphas_step, a_min=clip_value, a_max=1.0)
    alphas2 = np.cumprod(alphas_step, axis=0)

    return alphas2


def get_beta_schedule(
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
    num_diffusion_timesteps: int = 1000,
    kind: str = "cosine",
    nu: float = 1.0,
    plot: bool = False,
    **kwargs
):
    if kind == "quad":
        betas = (
            torch.linspace(
                beta_start**0.5,
                beta_end**0.5,
                num_diffusion_timesteps,
                dtype=torch.get_default_dtype(),
            )
            ** 2
        )
    elif kind == "sigmoid":
        betas = torch.linspace(-6, 6, num_diffusion_timesteps)
        betas = torch.sigmoid(betas) * (beta_end - beta_start) + beta_start
    elif kind == "linear":
        betas = torch.linspace(beta_start, beta_end, num_diffusion_timesteps)
    elif kind == "cosine":
        s = kwargs.get("s")
        if s is None:
            s = 0.008
        steps = num_diffusion_timesteps + 2
        x = torch.linspace(0, num_diffusion_timesteps, steps)
        alphas_cumprod = (
            torch.cos(((x / num_diffusion_timesteps) + s) / (1 + s) * torch.pi * 0.5)
            ** 2
        )
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clip(betas, 0.0, 0.999)
    elif kind == "polynomial":
        s = kwargs.get("s")
        p = kwargs.get("p")
        if s is None:
            s = 1e-4
        if p is None:
            p = 3.0
        steps = num_diffusion_timesteps + 1
        x = np.linspace(0, steps, steps)
        alphas2 = (1 - np.power(x / steps, p)) ** 2
        alphas2 = clip_noise_schedule(alphas2, clip_value=0.001)
        precision = 1 - 2 * s
        alphas2 = precision * alphas2 + s
        alphas = np.sqrt(alphas2)
        betas = 1.0 - alphas
        betas = torch.from_numpy(betas).float()
    elif kind == "adaptive":
        s = kwargs.get("s")
        if s is None:
            s = 0.008
        steps = num_diffusion_timesteps + 2
        x = np.linspace(0, steps, steps)
        x = np.expand_dims(x, 0)  # ((1, steps))

        nu_arr = np.array(nu)  # (components, )  # X, charges, E, y, pos

        alphas_cumprod = (
            np.cos(0.5 * np.pi * (((x / steps) ** nu_arr) + s) / (1 + s)) ** 2
        )  # ((components, steps))
        # divide every element of alphas_cumprod by the first element of alphas_cumprod
        alphas_cumprod_new = alphas_cumprod / alphas_cumprod[:, 0]
        # remove the first element of alphas_cumprod and then multiply every element by the one before it
        alphas = alphas_cumprod_new[:, 1:] / alphas_cumprod_new[:, :-1]

        betas = 1 - alphas  # ((components, steps)) # X, charges, E, y, pos
        betas = np.swapaxes(betas, 0, 1)
        betas = torch.clip(torch.from_numpy(betas), 0.0, 0.999).squeeze().float()

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
        nu: float = 1.0,
        enforce_zero_terminal_snr: bool = False,
        T: int = 500,
        param: str = "data",
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
        assert param in ["noise", "data"]
        self.param = param
        
        assert schedule in [
            "linear",
            "quad",
            "cosine",
            "sigmoid",
            "polynomial",
            "adaptive",
        ]

        discrete_betas = get_beta_schedule(
            beta_start=beta_min,
            beta_end=beta_max,
            num_diffusion_timesteps=N,
            kind=schedule,
            nu=nu,
            plot=False,
        )

        self.schedule = schedule
        self.T = T

        if enforce_zero_terminal_snr:
            discrete_betas = self.enforce_zero_terminal_snr(betas=discrete_betas)

        self.enforce_zero_terminal_snr = enforce_zero_terminal_snr

        sqrt_betas = torch.sqrt(discrete_betas)
        alphas = 1.0 - discrete_betas

        sqrt_alphas = torch.sqrt(alphas)
        if schedule == "adaptive":
            log_alpha = torch.log(alphas)
            log_alpha_bar = torch.cumsum(log_alpha, dim=0)
            alphas_cumprod = torch.exp(log_alpha_bar)
            log_alpha = torch.log(alphas)
            log_alpha_bar = torch.cumsum(log_alpha, dim=0)
            self._alphas = alphas
            self._log_alpha_bar = log_alpha_bar
            self._alphas_bar = torch.exp(log_alpha_bar)
            self._sigma2_bar = -torch.expm1(2 * log_alpha_bar)
            self._sigma_bar = torch.sqrt(self._sigma2_bar)
            self._gamma = (
                torch.log(-torch.special.expm1(2 * log_alpha_bar)) - 2 * log_alpha_bar
            )
        else:
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

    def enforce_zero_terminal_snr(self, betas):
        # Convert betas to alphas_bar_sqrt
        alphas = 1 - betas
        alphas_bar = alphas.cumprod(0)
        alphas_bar_sqrt = alphas_bar.sqrt()

        # Store old values.
        alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
        alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()

        # Shift so the last timestep is zero.
        alphas_bar_sqrt -= alphas_bar_sqrt_T

        # Scale so the first timestep is back to the old value.
        alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)

        # Convert alphas_bar_sqrt to betas
        alphas_bar = alphas_bar_sqrt**2
        alphas = alphas_bar[1:] / alphas_bar[:-1]
        alphas = torch.cat([alphas_bar[0:1], alphas])
        betas = 1 - alphas

        return betas

    def marginal_prob(self, x: Tensor, t: Tensor):
        """_summary_
        Eq. 4 in https://arxiv.org/abs/2006.11239
        Args:
            x (Tensor): _description_ Continuous data feature tensor
            t (Tensor): _description_ Discrete time variable between 1 and T
        Returns:
            _type_: _description_
        """

        assert str(t.dtype) == "torch.int64"
        expand_axis = len(x.size()) - 1

        if self.schedule == "adaptive":
            signal = self.get_alpha_bar(t_int=t)
            std = self.get_sigma_bar(t_int=t)
        else:
            signal = self.sqrt_alphas_cumprod[t]
            std = self.sqrt_1m_alphas_cumprod[t]

        for _ in range(expand_axis):
            signal = signal.unsqueeze(-1)
            std = std.unsqueeze(-1)

        mean = signal * x

        return mean, std

    def plot_signal_to_noise(self):
        t = torch.arange(0, self.N)
        signal = self.alphas_cumprod[t]
        noise = 1.0 - signal
        plt.plot(t, signal, label="signal")
        plt.plot(t, noise, label="noise")
        plt.xlabel("timesteps")
        plt.legend()
        plt.show()

    def get_alpha_bar(self, t_normalized=None, t_int=None, key=None):
        assert int(t_normalized is None) + int(t_int is None) == 1
        if t_int is None:
            t_int = torch.round(t_normalized * self.T)
        a = self._alphas_bar.to(t_int.device)[t_int.long()]
        return a.float()

    def get_sigma_bar(self, t_normalized=None, t_int=None, key=None):
        assert int(t_normalized is None) + int(t_int is None) == 1
        if t_int is None:
            t_int = torch.round(t_normalized * self.T)
        s = self._sigma_bar.to(t_int.device)[t_int]
        return s.float()

    def get_sigma2_bar(self, t_normalized=None, t_int=None, key=None):
        assert int(t_normalized is None) + int(t_int is None) == 1
        if t_int is None:
            t_int = torch.round(t_normalized * self.T)
        s = self._sigma2_bar.to(t_int.device)[t_int]
        return s.float()

    def get_gamma(self, t_normalized=None, t_int=None, key=None):
        assert int(t_normalized is None) + int(t_int is None) == 1
        if t_int is None:
            t_int = torch.round(t_normalized * self.T)
        g = self._gamma.to(t_int.device)[t_int]
        return g.float()

    def sigma_pos_ts_sq(self, t_int, s_int):
        gamma_s = self.get_gamma(t_int=s_int)
        gamma_t = self.get_gamma(t_int=t_int)
        delta_soft = F.softplus(gamma_s) - F.softplus(gamma_t)
        sigma_squared = -torch.expm1(delta_soft)
        return sigma_squared

    def get_alpha_pos_ts(self, t_int, s_int):
        log_a_bar = self._log_alpha_bar.to(t_int.device)
        ratio = torch.exp(log_a_bar[t_int] - log_a_bar[s_int])
        return ratio.float()

    def get_alpha_pos_ts_sq(self, t_int, s_int):
        log_a_bar = self._log_alpha_bar.to(t_int.device)
        ratio = torch.exp(2 * log_a_bar[t_int] - 2 * log_a_bar[s_int])
        return ratio.float()

    def get_sigma_pos_sq_ratio(self, s_int, t_int):
        log_a_bar = self._log_alpha_bar.to(t_int.device)
        s2_s = -torch.expm1(2 * log_a_bar[s_int])
        s2_t = -torch.expm1(2 * log_a_bar[t_int])
        ratio = torch.exp(torch.log(s2_s) - torch.log(s2_t))
        return ratio.float()

    def get_x_pos_prefactor(self, s_int, t_int):
        """a_s (s_t^2 - a_t_s^2 s_s^2) / s_t^2"""
        a_s = self.get_alpha_bar(t_int=s_int)
        alpha_ratio_sq = self.get_alpha_pos_ts_sq(t_int=t_int, s_int=s_int)
        sigma_ratio_sq = self.get_sigma_pos_sq_ratio(s_int=s_int, t_int=t_int)
        prefactor = a_s * (1 - alpha_ratio_sq * sigma_ratio_sq)
        return prefactor.float()

    def sample_reverse(
        self,
        t,
        xt,
        model_out,
        batch,
        cog_proj=False,
        edge_index_global=None,
        eta_ddim: float = 1.0
    ):  
        
        rev_sigma = self.reverse_posterior_sigma[t].unsqueeze(-1)
        noise = torch.randn_like(xt)
        std = rev_sigma[batch]
        
        if edge_index_global is not None:
            noise = 0.5 * (noise + noise.permute(1, 0, 2))
            noise = noise[edge_index_global[0, :], edge_index_global[1, :], :]
        else:
            bs = int(batch.max()) + 1
            if cog_proj:
                noise = zero_mean(noise, batch=batch, dim_size=bs, dim=0)
                                
        if self.param =="data":
            sigmast = self.sqrt_1m_alphas_cumprod[t].unsqueeze(-1)
            sigmas2t = sigmast.pow(2)

            sqrt_alphas = self.sqrt_alphas[t].unsqueeze(-1)
            sqrt_1m_alphas_cumprod_prev = torch.sqrt(
                (1.0 - self.alphas_cumprod_prev[t]).clamp_min(0.0)
            ).unsqueeze(-1)
            one_m_alphas_cumprod_prev = sqrt_1m_alphas_cumprod_prev.pow(2)
            sqrt_alphas_cumprod_prev = torch.sqrt(self.alphas_cumprod_prev[t].unsqueeze(-1))
            one_m_alphas = self.discrete_betas[t].unsqueeze(-1)

            mean = (
                sqrt_alphas[batch] * one_m_alphas_cumprod_prev[batch] * xt
                + sqrt_alphas_cumprod_prev[batch] * one_m_alphas[batch] * model_out
            )
            mean = (1.0 / sigmas2t[batch]) * mean
        
            xt_m1 = mean + eta_ddim * std * noise
            
        elif self.param == "noise":
            noise_predictor_prefactor = self.discrete_betas[t] / self.sqrt_1m_alphas_cumprod[t]
            noise_predictor_prefactor = noise_predictor_prefactor.unsqueeze(-1)
            factor = 1.0 / self.sqrt_alphas[t].unsqueeze(-1)
            
            xt_m1 = factor * (xt - noise_predictor_prefactor[batch] * model_out) + std * noise
            
        return xt_m1

    def sample_pos(self, t, pos, data_batch):
        # Coords: point cloud in R^3
        # sample noise for coords and recenter
        bs = int(data_batch.max()) + 1

        noise_coords_true = torch.randn_like(pos)
        noise_coords_true = zero_mean(
            noise_coords_true, batch=data_batch, dim_size=bs, dim=0
        )
        # get signal and noise coefficients for coords
        mean_coords, std_coords = self.marginal_prob(x=pos, t=t[data_batch])
        # perturb coords
        pos_perturbed = mean_coords + std_coords * noise_coords_true
        
        return pos_perturbed, noise_coords_true

    def sample_reverse_adaptive(
        self,
        s,
        t,
        xt,
        model_out,
        batch,
        cog_proj=False,
        edge_attrs=None,
        edge_index_global=None,
        eta_ddim: float = 1.0
    ):  
        
        if edge_index_global is not None:
            noise = torch.randn_like(edge_attrs)
            noise = 0.5 * (noise + noise.permute(1, 0, 2))
            noise = noise[edge_index_global[0, :], edge_index_global[1, :], :]
        else:
            bs = int(batch.max()) + 1
            noise = torch.randn_like(xt)
            if cog_proj:
                noise = zero_mean(noise, batch=batch, dim_size=bs, dim=0)
                
        sigma_sq_ratio = self.get_sigma_pos_sq_ratio(s_int=s, t_int=t)
       
        prefactor1 = self.get_sigma2_bar(t_int=t)
        prefactor2 = self.get_sigma2_bar(t_int=s) * self.get_alpha_pos_ts_sq(
            t_int=t, s_int=s
        )
        sigma2_t_s = prefactor1 - prefactor2
        noise_prefactor_sq = sigma2_t_s * sigma_sq_ratio
        noise_prefactor = torch.sqrt(noise_prefactor_sq).unsqueeze(-1)

        if self.param == "data":
            z_t_prefactor = (self.get_alpha_pos_ts(t_int=t, s_int=s) * sigma_sq_ratio).unsqueeze(-1)
            x_prefactor = self.get_x_pos_prefactor(s_int=s, t_int=t).unsqueeze(-1)
            mu = z_t_prefactor[batch] * xt + x_prefactor[batch] * model_out
            xt_m1 = mu + eta_ddim * noise_prefactor[batch] * noise
        elif self.param == "noise":
            alpha_ts = self.get_alpha_pos_ts(t_int=t, s_int=s).unsqueeze(-1)
            z_t_prefactor = 1.0 / alpha_ts
            sigma2_t = self.get_sigma2_bar(t_int=t).unsqueeze(-1)
            sigma2_s = self.get_sigma2_bar(t_int=s).unsqueeze(-1)
            sigma2_ts = sigma2_t - alpha_ts.pow(2) * sigma2_s
            noise_predictor_prefactor = sigma2_ts / (alpha_ts * sigma2_t.sqrt())
            xt_m1 = z_t_prefactor[batch] * xt - noise_predictor_prefactor[batch] * model_out \
                + noise_prefactor[batch] * noise
        return xt_m1
    
    def sample_reverse_ddim(
        self,
        t,
        xt,
        model_out,
        batch,
        cog_proj=False,
        edge_index_global=None,
        eta_ddim: float = 1.0
    ):
        assert 0.0 <= eta_ddim <= 1.0
        
        if self.schedule == 'cosine':
            rev_sigma = self.reverse_posterior_sigma[t].unsqueeze(-1)
            rev_sigma_ddim = eta_ddim * rev_sigma
            
            alphas_cumprod_prev = self.alphas_cumprod_prev[t].unsqueeze(-1)
            sqrt_alphas_cumprod_prev = alphas_cumprod_prev.sqrt()
            
            sqrt_alphas_cumprod = self.sqrt_alphas_cumprod[t].unsqueeze(-1)
            sqrt_one_m_alphas_cumprod = self.sqrt_1m_alphas_cumprod[t].unsqueeze(-1)
        elif self.schedule == 'adaptive':
            sigma_sq_ratio = self.get_sigma_pos_sq_ratio(s_int=t-1, t_int=t)
            prefactor1 = self.get_sigma2_bar(t_int=t)
            prefactor2 = self.get_sigma2_bar(t_int=t-1) * self.get_alpha_pos_ts_sq(
                t_int=t, s_int=t-1
            )
            sigma2_t_s = prefactor1 - prefactor2
            noise_prefactor_sq = sigma2_t_s * sigma_sq_ratio
            rev_sigma = torch.sqrt(noise_prefactor_sq).unsqueeze(-1)
            rev_sigma_ddim = eta_ddim * rev_sigma
            
            alphas_cumprod_prev = self.get_alpha_bar(t_int=t-1).unsqueeze(-1)
            sqrt_alphas_cumprod_prev = alphas_cumprod_prev.sqrt()
            sqrt_alphas_cumprod = self.get_alpha_bar(t_int=t).sqrt().unsqueeze(-1)
            sqrt_one_m_alphas_cumprod = (1.0 - self.get_alpha_bar(t_int=t).unsqueeze(-1)).clamp_min(0.0).sqrt()
            
        noise = torch.randn_like(xt)
             
        mean = (
            sqrt_alphas_cumprod_prev[batch] * model_out + \
                (1.0 - alphas_cumprod_prev - rev_sigma_ddim.pow(2)).clamp_min(0.0).sqrt()[batch] * \
                ( (xt - sqrt_alphas_cumprod[batch] * model_out) / sqrt_one_m_alphas_cumprod[batch] )
        )
        
        if edge_index_global is not None:
            noise = 0.5 * (noise + noise.permute(1, 0, 2))
            noise = noise[edge_index_global[0, :], edge_index_global[1, :], :]
        else:
            bs = int(batch.max()) + 1
            if cog_proj:
                noise = zero_mean(noise, batch=batch, dim_size=bs, dim=0)
 
        xt_m1 = mean + rev_sigma_ddim[batch] * noise

        return xt_m1

if __name__ == "__main__":
    T = 500
    schedule = "cosine"

    sde = DiscreteDDPM(
        beta_min=1e-4,
        beta_max=2e-2,
        N=T,
        scaled_reverse_posterior_sigma=True,
        schedule=schedule,
        enforce_zero_terminal_snr=False,
    )

    sde.plot_signal_to_noise()

    plt.plot(range(len(sde.discrete_betas)), sde.discrete_betas, label="betas")
    plt.plot(range(len(sde.alphas)), sde.alphas, label="alphas")
    plt.xlabel("t")
    plt.legend()
    plt.show()

    #### SNR
    # equation 1 and 2 in https://arxiv.org/pdf/2107.00630.pdf
    signal = sde.alphas_cumprod  # \alpha_t
    noise = 1.0 - signal  # \sigma_t^2
    snr = signal / noise
    plt.plot(range(len(signal)), snr, label="SNR(t)")
    plt.legend()
    plt.show()

    #### notation from kingma
    gamma = torch.log(snr)
    plt.plot(range(len(signal)), gamma, label="gamam(t) = log_e(SNR(t))")
    plt.legend()
    plt.show()

    ### From https://arxiv.org/pdf/2303.00848.pdf
    # sigmoidal
    plt.plot(gamma, torch.sigmoid(-gamma + 2), label="sigmoidal-weights")
    plt.legend()
    plt.show()

    plt.plot(gamma, torch.sigmoid(-gamma + 2), label="sigmoidal-weights")
    plt.legend()
    plt.show()

    plt.plot(
        torch.arange(len(gamma)), torch.sigmoid(-gamma + 2), label="sigmoidal-weights"
    )
    plt.legend()
    plt.show()
