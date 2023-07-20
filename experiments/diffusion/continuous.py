import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn


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
        betas = torch.clip(torch.from_numpy(betas), 0.0, 0.999).squeeze()

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

        if enforce_zero_terminal_snr:
            discrete_betas = self.enforce_zero_terminal_snr(betas=discrete_betas)

        self.enforce_zero_terminal_snr = enforce_zero_terminal_snr

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
        t = torch.arange(0, self.N)
        signal = self.alphas_cumprod[t]
        noise = 1.0 - signal
        plt.plot(t, signal, label="signal")
        plt.plot(t, noise, label="noise")
        plt.xlabel("timesteps")
        plt.legend()
        plt.show()


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
