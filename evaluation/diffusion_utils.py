import torch
import numpy as np
from torch_scatter import scatter_add, scatter_mean
import torch.nn.functional as F
import math
from csv import writer
import csv
import os
import pickle


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(
            current_model.parameters(), ma_model.parameters()
        ):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def sample_gaussian(size, device, batch=None, fix_noise=False):
    if fix_noise:
        bs = len(batch.unique())
        num_nodes = batch.bincount()[0]
        assert batch.bincount()[0] == batch.bincount()[1]
        size = (num_nodes, size[1])
        x = torch.randn(size, device=device).repeat(bs, 1)
    else:
        x = torch.randn(size, device=device)
    return x


def remove_mean(x, batch):
    mean = scatter_mean(x, batch, dim=0)
    x = x - mean[batch]
    return x


def remove_mean_pocket(x_lig, x_pocket, batch, batch_pocket):

    # Just subtract the center of mass of the sampled part
    mean = scatter_mean(x_lig, batch, dim=0)

    x_lig = x_lig - mean[batch]
    x_pocket = x_pocket - mean[batch_pocket]
    return x_lig, x_pocket


def assert_mean_zero(x, batch=None):
    if batch is not None:
        mean = scatter_mean(x, batch, dim=0)
    else:
        mean = x.mean()
    if mean.abs().max().item() > 1e-4:
        print(f"Position shift: {mean.abs().max().item()}\n")


def sample_center_gravity_zero_gaussian(size, device, batch, fix_noise=False):
    assert len(size) == 2

    if fix_noise:
        bs = len(batch.unique())
        num_nodes = batch.bincount()[0]
        assert batch.bincount()[0] == batch.bincount()[1]
        size = (num_nodes, 3)
        x = torch.randn(size, device=device).repeat(bs, 1)
    else:
        x = torch.randn(size, device=device)
    # This projection only works because Gaussian is rotation invariant around
    # zero and samples are independent!
    x_projected = remove_mean(x, batch)
    return x_projected


def center_gravity_zero_gaussian_log_likelihood(x, batch):
    assert len(x.size()) == 2
    N_embedded, D = x.size()
    assert_mean_zero(x, batch)

    # r is invariant to a basis change in the relevant hyperplane, the masked
    # out values will have zero contribution.
    r2 = torch.sum(scatter_add(x.pow(2), batch, dim=0), dim=1)
    # The relevant hyperplane is (N-1) * D dimensional.
    N = batch.bincount()  # N has shape [B]
    degrees_of_freedom = (N - 1) * D

    # Normalizing constant and logpx are computed:
    log_normalizing_constant = -0.5 * degrees_of_freedom * np.log(2 * np.pi)
    log_px = -0.5 * r2 + log_normalizing_constant

    return log_px


def standard_gaussian_log_likelihood(x, batch):
    # Normalizing constant and logpx are computed:
    log_px_elementwise = torch.sum(
        scatter_add(-0.5 * x * x - 0.5 * np.log(2 * np.pi), batch, dim=0), dim=1
    )
    return log_px_elementwise


# Defining some useful util functions.
def expm1(x: torch.Tensor) -> torch.Tensor:
    return torch.expm1(x)


def softplus(x: torch.Tensor) -> torch.Tensor:
    return F.softplus(x)


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


def polynomial_schedule(timesteps: int, s=1e-4, power=3.0):
    """
    A noise schedule based on a simple polynomial equation: 1 - x^power.
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas2 = (1 - np.power(x / steps, power)) ** 2

    alphas2 = clip_noise_schedule(alphas2, clip_value=0.001)

    precision = 1 - 2 * s

    alphas2 = precision * alphas2 + s

    return alphas2


def cosine_beta_schedule(timesteps, s=0.008, raise_to_power: float = 1):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = np.clip(betas, a_min=0, a_max=0.999)
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)

    if raise_to_power != 1:
        alphas_cumprod = np.power(alphas_cumprod, raise_to_power)

    return alphas_cumprod


def gaussian_entropy(mu, sigma, batch):
    # In case sigma needed to be broadcast (which is very likely in this code).
    zeros = torch.zeros_like(mu)
    return torch.sum(
        scatter_add(
            (zeros + 0.5 * torch.log(2 * np.pi * sigma**2) + 0.5), batch, dim=0
        ),
        dim=1,
    )


def gaussian_KL(q_mu, q_sigma, p_mu, p_sigma, batch):
    """Computes the KL distance between two normal distributions.

    Args:
        q_mu: Mean of distribution q.
        q_sigma: Standard deviation of distribution q.
        p_mu: Mean of distribution p.
        p_sigma: Standard deviation of distribution p.
    Returns:
        The KL distance, summed over all dimensions except the batch dim.
    """
    return torch.sum(
        scatter_add(
            (
                torch.log(p_sigma / q_sigma)
                + 0.5 * (q_sigma**2 + (q_mu - p_mu) ** 2) / (p_sigma**2)
                - 0.5
            ),
            batch,
            dim=0,
        ),
        dim=1,
    )


def gaussian_KL_edge(q_mu, q_sigma, p_mu, p_sigma, batch):
    """Computes the KL distance between two normal distributions.

    Args:
        q_mu: Mean of distribution q.
        q_sigma: Standard deviation of distribution q.
        p_mu: Mean of distribution p.
        p_sigma: Standard deviation of distribution p.
    Returns:
        The KL distance, summed over all dimensions except the batch dim.
    """
    return torch.sum(
        scatter_add(
            (
                torch.log(p_sigma / q_sigma)
                + 0.5 * (q_sigma**2 + (q_mu - p_mu) ** 2) / (p_sigma**2)
                - 0.5
            ),
            batch,
            dim=0,
        ),
        dim=1,
    )


def gaussian_KL_for_dimension(q_mu, q_sigma, p_mu, p_sigma, d, batch):
    """Computes the KL distance between two normal distributions.

    Args:
        q_mu: Mean of distribution q.
        q_sigma: Standard deviation of distribution q.
        p_mu: Mean of distribution p.
        p_sigma: Standard deviation of distribution p.
    Returns:
        The KL distance, summed over all dimensions except the batch dim.
    """
    mu_norm2 = torch.sum(scatter_add((q_mu - p_mu) ** 2, batch, dim=0), dim=1)
    assert len(q_sigma.size()) == 1
    assert len(p_sigma.size()) == 1
    return (
        d * torch.log(p_sigma / q_sigma)
        + 0.5 * (d * q_sigma**2 + mu_norm2) / (p_sigma**2)
        - 0.5 * d
    )


def gaussian_KL_pocket(q_mu_minus_p_mu_squared, q_sigma, p_sigma, d):
    """Computes the KL distance between two normal distributions.
    Args:
        q_mu_minus_p_mu_squared: Squared difference between mean of
            distribution q and distribution p: ||mu_q - mu_p||^2
        q_sigma: Standard deviation of distribution q.
        p_sigma: Standard deviation of distribution p.
        d: dimension
    Returns:
        The KL distance
    """
    return (
        d * torch.log(p_sigma / q_sigma)
        + 0.5 * (d * q_sigma**2 + q_mu_minus_p_mu_squared) / (p_sigma**2)
        - 0.5 * d
    )


class PositiveLinear(torch.nn.Module):
    """Linear layer with weights forced to be positive."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        weight_init_offset: int = -2,
    ):
        super(PositiveLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.empty((out_features, in_features)))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.weight_init_offset = weight_init_offset
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        with torch.no_grad():
            self.weight.add_(self.weight_init_offset)

        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        positive_weight = softplus(self.weight)
        return F.linear(input, positive_weight, self.bias)


class SinusoidalPosEmb(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        x = x.squeeze() * 1000
        assert len(x.shape) == 1
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class PredefinedNoiseSchedule(torch.nn.Module):
    """
    Predefined noise schedule. Essentially creates a lookup array for predefined (non-learned) noise schedules.
    """

    def __init__(self, noise_schedule, timesteps, precision):
        super(PredefinedNoiseSchedule, self).__init__()
        self.timesteps = timesteps

        if noise_schedule == "cosine":
            alphas2 = cosine_beta_schedule(timesteps)
        elif "polynomial" in noise_schedule:
            splits = noise_schedule.split("_")
            assert len(splits) == 2
            power = float(splits[1])
            alphas2 = polynomial_schedule(timesteps, s=precision, power=power)
        else:
            raise ValueError(noise_schedule)

        print("alphas2", alphas2)

        sigmas2 = 1 - alphas2

        log_alphas2 = np.log(alphas2)
        log_sigmas2 = np.log(sigmas2)

        log_alphas2_to_sigmas2 = log_alphas2 - log_sigmas2

        print("gamma", -log_alphas2_to_sigmas2)

        self.gamma = torch.nn.Parameter(
            torch.from_numpy(-log_alphas2_to_sigmas2).float(), requires_grad=False
        )

    def forward(self, t):
        t_int = torch.round(t * self.timesteps).long()
        return self.gamma[t_int]


class GammaNetwork(torch.nn.Module):
    """The gamma network models a monotonic increasing function. Construction as in the VDM paper."""

    def __init__(self):
        super().__init__()

        self.l1 = PositiveLinear(1, 1)
        self.l2 = PositiveLinear(1, 1024)
        self.l3 = PositiveLinear(1024, 1)

        self.gamma_0 = torch.nn.Parameter(torch.tensor([-5.0]))
        self.gamma_1 = torch.nn.Parameter(torch.tensor([10.0]))
        self.show_schedule()

    def show_schedule(self, num_steps=50):
        t = torch.linspace(0, 1, num_steps).view(num_steps, 1)
        gamma = self.forward(t)
        print("Gamma schedule:")
        print(gamma.detach().cpu().numpy().reshape(num_steps))

    def gamma_tilde(self, t):
        l1_t = self.l1(t)
        return l1_t + self.l3(torch.sigmoid(self.l2(l1_t)))

    def forward(self, t):
        zeros, ones = torch.zeros_like(t), torch.ones_like(t)
        # Not super efficient.
        gamma_tilde_0 = self.gamma_tilde(zeros)
        gamma_tilde_1 = self.gamma_tilde(ones)
        gamma_tilde_t = self.gamma_tilde(t)

        # Normalize to [0, 1]
        normalized_gamma = (gamma_tilde_t - gamma_tilde_0) / (
            gamma_tilde_1 - gamma_tilde_0
        )

        # Rescale to [gamma_0, gamma_1]
        gamma = self.gamma_0 + (self.gamma_1 - self.gamma_0) * normalized_gamma

        return gamma


def cdf_standard_gaussian(x):
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2)))


def remove_mean_no_mask(x):
    mean = torch.mean(x, dim=1, keepdim=True)
    x = x - mean
    return x


def remove_mean_with_mask(x, node_mask):
    masked_max_abs_value = (x * (1 - node_mask)).abs().sum().item()
    assert masked_max_abs_value < 1e-5, f"Error {masked_max_abs_value} too high"
    N = node_mask.sum(1, keepdims=True)

    mean = torch.sum(x, dim=1, keepdim=True) / N
    x = x - mean * node_mask
    return x


def reverse_tensor(x):
    return x[torch.arange(x.size(0) - 1, -1, -1)]


def save_log(log, path):
    with open(path, "wb") as f:
        pickle.dump(log, f)


def load_log(path):
    with open(path, "rb") as f:
        b = pickle.load(f)
    return b


def write_to_csv(validity_dict, rdkit_tuple, midi_log, path):
    if os.path.exists(path):
        with open(path, "a") as f_object:
            writer_object = writer(f_object)
            writer_object.writerow(
                [
                    validity_dict["mol_stable"],
                    validity_dict["atm_stable"],
                    rdkit_tuple[0][0],
                    rdkit_tuple[0][1],
                    rdkit_tuple[0][2],
                    midi_log["sampling/NumNodesW1"],
                    midi_log["sampling/AtomTypesTV"],
                    midi_log["sampling/ChargeW1"],
                    midi_log["sampling/EdgeTypesTV"],
                    midi_log["sampling/ValencyW1"],
                    midi_log["sampling/BondLengthsW1"],
                    midi_log["sampling/AnglesW1"],
                ]
            )
            f_object.close()
    else:
        fields = [
            "Molecule stability",
            "Atom stability",
            "Validity",
            "Uniqueness",
            "Novelty",
            "W1NumNodes",
            "TVAtomTypes",
            "W1Charges",
            "TVEdgeTypes",
            "W1Valency",
            "W1BondLengths",
            "W1Angles",
        ]
        with open(path, "w") as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(fields)
            csvwriter.writerow(
                [
                    validity_dict["mol_stable"],
                    validity_dict["atm_stable"],
                    rdkit_tuple[0][0],
                    rdkit_tuple[0][1],
                    rdkit_tuple[0][2],
                    midi_log["sampling/NumNodesW1"],
                    midi_log["sampling/AtomTypesTV"],
                    midi_log["sampling/ChargeW1"],
                    midi_log["sampling/EdgeTypesTV"],
                    midi_log["sampling/ValencyW1"],
                    midi_log["sampling/BondLengthsW1"],
                    midi_log["sampling/AnglesW1"],
                ]
            )
