import logging
import os
import pdb
from argparse import ArgumentParser
from typing import List, Optional, Tuple
import torch.nn.functional as F
import time
import torch_geometric
import pickle
from callbacks.ema import ExponentialMovingAverage
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
    TQDMProgressBar,
)

from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np
from torch import Tensor
from torch_geometric.data import Batch
from torch_geometric.typing import OptTensor
from torch_geometric.utils import dense_to_sparse, sort_edge_index

from torch_cluster import radius_graph
from torch_sparse import coalesce
from torch_scatter import scatter_mean, scatter_add
from evaluation.diffusion_utils import (
    PredefinedNoiseSchedule,
    GammaNetwork,
    assert_mean_zero,
    remove_mean,
    cdf_standard_gaussian,
    gaussian_KL,
    gaussian_KL_edge,
    gaussian_KL_for_dimension,
    expm1,
    softplus,
    sample_center_gravity_zero_gaussian,
    sample_gaussian,
    write_to_csv_full,
    save_log,
    reverse_tensor,
)
import evaluation.diffusion_visualiser as vis
from evaluation.diffusion_analyze import (
    sanitize_molecules_openbabel,
)
from config_file import get_dataset_info
from evaluation.diffusion_distribution import (
    prepare_context,
    get_distributions,
)
from qm9.analyze_sampling import SamplingMetrics
from qm9.info_data import QM9Infos
from qm9.utils_metrics import batch_to_list
from qm9.utils_sampling import (
    sanitize_molecules_openbabel,
    check_stability_given_bonds,
    Molecule,
)
from qm9.utils import PlaceHolder
from qm9.visualize_bonds import visualize_mol, visualize_chain, visualize

from e3moldiffusion.molfeat import get_bond_feature_dims
from e3moldiffusion.coordsatomsbonds import ScoreAtomsBondsModel
from e3moldiffusion.et_model import create_model

logging.getLogger("lightning").setLevel(logging.WARNING)

BOND_FEATURE_DIMS = get_bond_feature_dims()[0]


class Trainer(pl.LightningModule):
    def __init__(
        self,
        hparams,
        dataset_info=None,
        dataset_statistics=None,
        train_smiles=None,
        properties_norm=None,
        nodes_dist=None,
        prop_dist=None,
    ):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.i = 0

        self.properties_norm = properties_norm

        self.dataset_info = dataset_info
        self.train_smiles = train_smiles
        self.nodes_dist = nodes_dist
        self.prop_dist = prop_dist

        self.val_sampling_metrics = SamplingMetrics(
            self.train_smiles, dataset_statistics, test=False
        )

        assert self.hparams.loss_type in {"vlb", "l2"}
        self.loss_type = self.hparams.loss_type

        self.include_charges = self.hparams.include_charges

        if self.hparams.noise_schedule == "learned":
            assert self.loss_type == "vlb", (
                "A noise schedule can only be learned" " with a vlb objective."
            )

        # Only supported parametrization.
        assert self.hparams.parametrization == "eps"

        if self.hparams.noise_schedule == "learned":
            self.gamma = GammaNetwork()
        else:
            self.gamma = PredefinedNoiseSchedule(
                self.hparams.noise_schedule,
                timesteps=self.hparams.num_diffusion_timesteps,
                precision=self.hparams.noise_precision,
            )

        self.num_atom_features = self.hparams.num_atom_types + int(self.include_charges)
        self.num_bond_classes = 5
        self.n_dims = 3  # positions are always 3D
        self.num_context_features = self.hparams.num_context_features
        self.num_classes = self.num_atom_features - int(self.include_charges)

        self.T = self.hparams.num_diffusion_timesteps

        self.parametrization = self.hparams.parametrization

        self.norm_values = self.hparams.norm_values
        self.norm_biases = self.hparams.norm_biases
        self.register_buffer("buffer", torch.zeros(1))

        if self.hparams.noise_schedule != "learned":
            self.check_issues_norm_values()

        if self.hparams.model == "equivariant-transformer":
            self.model = create_model(self.hparams)

        elif self.hparams.model == "eqgat":
            self.model = ScoreAtomsBondsModel(
                hn_dim=(hparams["sdim"], hparams["vdim"]),
                num_layers=hparams["num_layers"],
                use_norm=hparams["use_norm"],
                use_cross_product=hparams["use_cross_product"],
                num_atom_features=self.num_atom_features,
                rbf_dim=hparams["num_rbf"],
                cutoff_local=hparams["cutoff_upper"],
                num_context_features=hparams["num_context_features"],
                vector_aggr="mean",
                local_global_model=hparams["fully_connected_layer"],
                use_local_edge_attr=hparams["use_local_edge_attr"],
                fully_connected=hparams["fully_connected"],
            )
        else:
            raise ValueError(f"Unknown architecture: {self.hparams.model}")

        print(f"\nRun with fully connected layer: {hparams['fully_connected_layer']}\n")
        print(f"\nDiffusion of charges: {hparams['include_charges']}\n")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams["lr"])
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            patience=self.hparams["lr_patience"],
            cooldown=self.hparams["lr_cooldown"],
            factor=self.hparams["lr_factor"],
        )
        scheduler = {
            "scheduler": lr_scheduler,
            "interval": "epoch",
            "frequency": self.hparams["lr_frequency"],
            "monitor": "val/loss",
            "strict": False,
        }
        return [optimizer], [scheduler]

    def forward(
        self,
        z,
        t,
        pos,
        batch,
        context,
        edge_index_local,
        edge_index_global,
        edge_attr_local,
        edge_attr_global,
    ):
        return self.model(
            z=z,
            t=t,
            pos=pos,
            batch=batch,
            context=context,
            edge_index_local=edge_index_local,
            edge_index_global=edge_index_global,
            edge_attr_local=edge_attr_local,
            edge_attr_global=edge_attr_global,
        )

    def get_fully_connected_edge_attr(self, batch):
        bond_edge_index = batch.bond_index
        bond_edge_attr = batch.bond_attr
        try:
            fc_edge_index = batch.fc_edge_index
        except:
            fc_edge_index = (
                torch.eq(batch.batch.unsqueeze(0), batch.batch.unsqueeze(-1))
                .int()
                .fill_diagonal_(0)
            )
            fc_edge_index, _ = dense_to_sparse(fc_edge_index)
            fc_edge_index = sort_edge_index(fc_edge_index, sort_by_row=False)
        n = batch.num_nodes

        fc_edge_attr = torch.full(
            size=(fc_edge_index.size(-1),),
            device=fc_edge_index.device,
            dtype=torch.long,
            fill_value=0,
        )
        fc_edge_index = torch.cat([fc_edge_index, bond_edge_index], dim=-1)
        fc_edge_attr = torch.cat([fc_edge_attr, bond_edge_attr], dim=0)
        fc_edge_index, fc_edge_attr = coalesce(
            index=fc_edge_index, value=fc_edge_attr, m=n, n=n, op="max"
        )
        fc_edge_index = sort_edge_index(fc_edge_index, sort_by_row=False)

        dense_edge = torch.zeros(n, n, dtype=torch.long, device=fc_edge_index.device)
        dense_edge[fc_edge_index[0, :], fc_edge_index[1, :]] = fc_edge_attr
        dense_edge_oh = (
            F.one_hot(dense_edge.view(-1, 1), num_classes=5).view(n, n, -1).float()
        )
        return fc_edge_index, dense_edge_oh

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")

    def validation_step(self, batch, batch_idx, *args):
        if len(args) == 0 or (len(args) > 0 and args[0] == 0):
            # validation step
            return self.step(batch, "val")
        # test step
        return self.step(batch, "test")

    def test_step(self, batch, batch_idx):
        return self.step(batch, "test")

    def step(self, batch: Batch, stage: str):
        node_feat: Tensor = batch.z
        pos: Tensor = batch.pos
        charges: Tensor = batch.charges
        data_batch: Tensor = batch.batch

        (
            edge_index_global,
            dense_edge_oh,
        ) = self.get_fully_connected_edge_attr(batch)

        bs = int(data_batch.max()) + 1

        if self.properties_norm is not None:
            context = prepare_context(
                self.hparams["properties_list"], batch, self.properties_norm
            )
        else:
            context = None

        z = F.one_hot(
            node_feat.squeeze().long(), num_classes=max(self.hparams["atom_types"]) + 1
        ).float()[:, self.hparams["atom_types"]]

        pos = remove_mean(pos, data_batch)
        charges = (
            torch.zeros(0)
            if not self.hparams["include_charges"]
            else charges.unsqueeze(1)
        ).to(node_feat.device)

        h = {"categorical": z, "integer": charges}

        # Normalize data
        pos, h, dense_edge_oh, delta_log_px = self.normalize(
            pos, h, dense_edge_oh, data_batch
        )
        # Reset delta_log_px if not vlb objective.
        if self.training and self.loss_type == "l2":
            delta_log_px = torch.zeros_like(delta_log_px)

        t0_always = stage != "train"
        loss, loss_dict = self.compute_loss(
            pos,
            h,
            data_batch=data_batch,
            edge_index_global=edge_index_global,
            dense_edge_oh=dense_edge_oh,
            context=context,
            t0_always=t0_always,
        )
        # Correct for normalization on x.
        assert loss.size() == delta_log_px.size()
        loss = loss - delta_log_px

        N = data_batch.bincount()
        log_pN = self.nodes_dist.log_prob(N)

        assert loss.size() == log_pN.size()
        loss = loss - log_pN
        loss = loss.mean(0)
        reg_term = torch.tensor([0.0]).to(loss.device)

        loss = loss + self.hparams["ode_regularization"] * reg_term
        loss = loss.squeeze()

        self.log_loss(loss=loss, loss_dict=loss_dict, batch_size=bs, stage=stage)

        return loss

    def log_loss(self, loss, loss_dict, batch_size, stage):
        if stage == "train":
            self.log(
                f"train/loss",
                loss,
                on_step=True,
                batch_size=batch_size,
            )
        else:
            self.log(
                f"{stage}/loss",
                loss,
                batch_size=batch_size,
                sync_dist=self.hparams["gpus"] > 1,
            )
        for key, value in loss_dict.items():
            if stage == "train":
                self.log(
                    f"train/{key}",
                    value.mean(0),
                    on_step=True,
                    batch_size=batch_size,
                )
            else:
                self.log(
                    f"{stage}/{key}",
                    value.mean(0),
                    batch_size=batch_size,
                    sync_dist=self.hparams["gpus"] > 1,
                )

    def validation_epoch_end(self, validation_step_outputs):
        if (self.current_epoch + 1) % self.hparams["test_interval"] == 0:
            i = self.i

            self.analyze_and_save(n_samples=self.hparams["n_stability_samples"])

            start = time.time()
            if self.properties_norm is not None:
                self.save_and_sample_conditional(
                    f"{self.hparams['log_dir']}/epoch_{self.current_epoch}_{i}/chain_conditional/"
                )

            self.save_and_sample_chain(
                f"{self.hparams['log_dir']}/epoch_{self.current_epoch}_{i}/chain/"
            )

            self.sample_different_sizes_and_save(
                f"{self.hparams['log_dir']}/epoch_{self.current_epoch}_{i}/"
            )

            print(f"Sampling took {time.time() - start:.2f} seconds")

            vis.visualize(
                f"{self.hparams['log_dir']}/epoch_{self.current_epoch}_{i}/",
                dataset_info=self.dataset_info,
                wandb=None,
            )
            vis.visualize_chain(
                f"{self.hparams['log_dir']}/epoch_{self.current_epoch}_{i}/chain/",
                self.dataset_info,
                wandb=None,
            )
            if self.properties_norm is not None:
                vis.visualize_chain(
                    f"{self.hparams['log_dir']}/epoch_{self.current_epoch}_{i}/chain_conditional/",
                    self.dataset_info,
                    wandb=None,
                    mode="conditional",
                )
            self.i += 1

    # ----------------------DIFFUSION UTILS----------------------#

    def check_issues_norm_values(self, num_stdevs=8):
        zeros = torch.zeros((1, 1))
        gamma_0 = self.gamma(zeros)
        sigma_0 = self.sigma(gamma_0).item()

        # Checked if 1 / norm_value is still larger than 10 * standard
        # deviation.
        max_norm_value = max(self.norm_values[1], self.norm_values[2])

        if sigma_0 * num_stdevs > 1.0 / max_norm_value:
            raise ValueError(
                f"Value for normalization value {max_norm_value} probably too "
                f"large with sigma_0 {sigma_0:.5f} and "
                f"1 / norm_value = {1. / max_norm_value}"
            )

    def network(
        self,
        x,
        t,
        data_batch,
        edge_index_local,
        edge_index_global,
        edge_attr_global,
        context=None,
    ):
        z = x[:, self.n_dims :]
        pos = x[:, : self.n_dims]

        edge_attr_local = (
            edge_attr_global[edge_index_local, :]
            if self.hparams["use_local_edge_attr"]
            else None
        )

        with torch.set_grad_enabled(self.training or self.hparams["energy_preserving"]):
            net_out = self(
                z=z,
                t=t,
                pos=pos,
                batch=data_batch,
                context=context,
                edge_index_local=edge_index_local,
                edge_index_global=edge_index_global,
                edge_attr_global=edge_attr_global,
                edge_attr_local=edge_attr_local,
            )

            noise_pred = remove_mean(net_out["score_coords"], data_batch)
            if self.training:
                assert torch.isnan(noise_pred).any() == False, "NaN noise predictions!"
                assert torch.isinf(noise_pred).any() == False, "Inf noise predictions!"
                assert_mean_zero(noise_pred, data_batch)

            edge_pred = net_out["score_bonds"]
            atoms_pred = net_out["score_atoms"]
            net_out = torch.cat([noise_pred, atoms_pred], dim=1)

            return net_out, edge_pred

    def sigma(self, gamma):
        """Computes sigma given gamma."""
        return torch.sqrt(torch.sigmoid(gamma))

    def alpha(self, gamma):
        """Computes alpha given gamma."""
        return torch.sqrt(torch.sigmoid(-gamma))

    def SNR(self, gamma):
        """Computes signal to noise ratio (alpha^2/sigma^2) given gamma."""
        return torch.exp(-gamma)

    def subspace_dimensionality(self, batch):
        """Compute the dimensionality on translation-invariant linear subspace where distributions on x are defined."""
        number_of_nodes = batch.bincount()
        return (number_of_nodes - 1) * self.n_dims

    def normalize(self, x, h, dense_edge_oh, batch):
        x = x / self.norm_values[0]
        delta_log_px = -self.subspace_dimensionality(batch) * np.log(
            self.norm_values[0]
        )

        # Casting to float in case h still has long or int type.
        h_cat = (h["categorical"].float() - self.norm_biases[1]) / self.norm_values[1]
        h_int = (h["integer"].float() - self.norm_biases[2]) / self.norm_values[2]

        # Create new h dictionary.
        h = {"categorical": h_cat, "integer": h_int}

        dense_edge_oh = (dense_edge_oh - self.norm_biases[3]) / self.norm_values[3]
        return x, h, dense_edge_oh, delta_log_px

    def unnormalize(self, x, h_cat, h_int, dense_edge_oh):
        x = x * self.norm_values[0]
        h_cat = h_cat * self.norm_values[1] + self.norm_biases[1]

        if self.include_charges:
            h_int = h_int * self.norm_values[2] + self.norm_biases[2]

        dense_edge_oh = dense_edge_oh * self.norm_values[3] + self.norm_biases[3]

        return x, h_cat, h_int, dense_edge_oh

    def unnormalize_z(self, z, dense_edge_oh):
        # Parse from z
        x, h_cat = (
            z[:, : self.n_dims],
            z[:, self.n_dims : self.n_dims + self.num_classes],
        )
        h_int = z[
            :, self.n_dims + self.num_classes : self.n_dims + self.num_classes + 1
        ]
        assert h_int.size(1) == self.include_charges

        # Unnormalize
        x, h_cat, h_int, dense_edge_oh = self.unnormalize(
            x,
            h_cat,
            h_int,
            dense_edge_oh,
        )
        output = torch.cat([x, h_cat, h_int], dim=1)
        return output, dense_edge_oh

    def sigma_and_alpha_t_given_s(
        self, gamma_t: torch.Tensor, gamma_s: torch.Tensor, batch: torch.Tensor
    ):
        """
        Computes sigma t given s, using gamma_t and gamma_s. Used during sampling.

        These are defined as:
            alpha t given s = alpha t / alpha s,
            sigma t given s = sqrt(1 - (alpha t given s) ^2 ).
        """

        sigma2_t_given_s = -expm1(softplus(gamma_s) - softplus(gamma_t))[batch]

        # alpha_t_given_s = alpha_t / alpha_s
        log_alpha2_t = F.logsigmoid(-gamma_t)
        log_alpha2_s = F.logsigmoid(-gamma_s)
        log_alpha2_t_given_s = log_alpha2_t - log_alpha2_s

        alpha_t_given_s = torch.exp(0.5 * log_alpha2_t_given_s)
        alpha_t_given_s = alpha_t_given_s[batch]

        sigma_t_given_s = torch.sqrt(sigma2_t_given_s)

        return sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s

    def kl_prior(self, xh, edge_out, batch, source):
        """Computes the KL between q(z1 | x) and the prior p(z1) = Normal(0, 1).

        This is essentially a lot of work for something that is in practice negligible in the loss. However, you
        compute it so that you see it when you've made a mistake in your noise schedule.
        """
        # Compute the last alpha value, alpha_T.
        ones = torch.ones((len(batch.unique()), 1), device=batch.device)
        gamma_T = self.gamma(ones)
        alpha_T = self.alpha(gamma_T)[batch]

        # Compute means.
        mu_T = alpha_T * xh
        mu_T_x, mu_T_h = mu_T[:, : self.n_dims], mu_T[:, self.n_dims :]
        mu_T_e = alpha_T[source] * edge_out

        sigma_T_x = self.sigma(gamma_T).squeeze()
        sigma_T_h = self.sigma(gamma_T)[batch]
        sigma_T_e = sigma_T_h[source]

        # Compute KL for h-part.
        zeros, ones = torch.zeros_like(mu_T_h), torch.ones_like(sigma_T_h)
        kl_distance_h = gaussian_KL(mu_T_h, sigma_T_h, zeros, ones, batch)

        zeros, ones = torch.zeros_like(mu_T_e), torch.ones_like(sigma_T_e)
        kl_distance_e = gaussian_KL_edge(mu_T_e, sigma_T_e, zeros, ones, batch, source)

        # Compute KL for x-part.
        zeros, ones = torch.zeros_like(mu_T_x), torch.ones_like(sigma_T_x)
        subspace_d = self.subspace_dimensionality(batch)
        kl_distance_x = gaussian_KL_for_dimension(
            mu_T_x, sigma_T_x, zeros, ones, d=subspace_d, batch=batch
        )
        return kl_distance_x + kl_distance_h + kl_distance_e

    def compute_x_pred(
        self, net_out, edge_out, z_t, edge_t, gamma_t, data_batch, source
    ):
        """Commputes x_pred, i.e. the most likely prediction of x."""
        if self.parametrization == "x":
            x_pred = net_out
            bond_pred = edge_out
        elif self.parametrization == "eps":
            sigma_t = self.sigma(gamma_t)[data_batch]
            alpha_t = self.alpha(gamma_t)[data_batch]
            eps_t = net_out
            x_pred = 1.0 / alpha_t * (z_t - sigma_t * eps_t)
            bond_pred = 1.0 / alpha_t[source] * (edge_t - sigma_t[source] * edge_out)
        else:
            raise ValueError(self.parametrization)

        return x_pred, bond_pred

    def compute_error(
        self,
        net_out,
        edge_out,
        noise_true,
        noise_edge_true,
        batch,
        source,
    ):
        """Computes error, i.e. the most likely prediction of x."""
        batch_size = len(batch.unique())
        coords_pred, coords_true = (
            net_out[:, : self.n_dims],
            noise_true[:, : self.n_dims],
        )
        atoms_pred, atoms_true = (
            net_out[:, self.n_dims : -1]
            if self.include_charges
            else net_out[:, self.n_dims :],
            noise_true[:, self.n_dims : -1]
            if self.include_charges
            else noise_true[:, self.n_dims :],
        )
        charges_pred, charges_true = (
            net_out[:, -1:] if self.include_charges else None,
            noise_true[:, -1:] if self.include_charges else None,
        )
        bonds_pred, bonds_true = edge_out, noise_edge_true

        coords_loss = torch.pow(coords_pred - coords_true, 2).sum(-1)
        coords_loss = scatter_mean(coords_loss, index=batch, dim=0, dim_size=batch_size)
        atoms_loss = torch.pow(atoms_pred - atoms_true, 2).mean(-1)
        atoms_loss = scatter_mean(atoms_loss, index=batch, dim=0, dim_size=batch_size)
        if self.include_charges:
            charges_loss = torch.pow(charges_pred - charges_true, 2).mean(-1)
            charges_loss = scatter_mean(
                charges_loss, index=batch, dim=0, dim_size=batch_size
            )
        else:
            charges_loss = torch.zeros_like(coords_loss, device=net_out.device)

        bonds_loss = torch.pow(bonds_pred - bonds_true, 2).mean(-1)
        bonds_loss = scatter_mean(bonds_loss, index=batch[source], dim=0)

        loss_dict = {
            "coords_loss": coords_loss,
            "atoms_loss": atoms_loss,
            "charges_loss": charges_loss,
            "bonds_loss": bonds_loss,
        }
        return loss_dict

    def log_constants_p_x_given_z0(self, batch):
        """Computes p(x|z0)."""
        batch_size = len(batch.unique())

        n_nodes = batch.bincount()  # N has shape [B]
        assert n_nodes.size() == (batch_size,)
        degrees_of_freedom_x = (n_nodes - 1) * self.n_dims

        zeros = torch.zeros((n_nodes.size(0), 1), device=batch.device)
        gamma_0 = self.gamma(zeros)

        # Recall that sigma_x = sqrt(sigma_0^2 / alpha_0^2) = SNR(-0.5 gamma_0).
        log_sigma_x = 0.5 * gamma_0.view(batch_size)

        return degrees_of_freedom_x * (-log_sigma_x - 0.5 * np.log(2 * np.pi))

    def sample_p_xh_given_z0(
        self,
        z_0,
        data_batch,
        edge_index_local,
        edge_index_global,
        edge_attr_global_t0,
        sample_size,
        context=None,
        fix_noise=False,
    ):
        source = edge_index_global[0]
        """Samples x ~ p(x|z0)."""
        zeros = torch.zeros(size=(z_0.size(0), 1), device=z_0.device)
        gamma_0 = self.gamma(zeros)
        # Computes sqrt(sigma_0^2 / alpha_0^2)
        sigma_x = self.SNR(-0.5 * gamma_0)

        eps_t0, eps_edge_t0 = self.network(
            x=z_0,
            t=zeros,
            data_batch=data_batch,
            edge_index_local=edge_index_local,
            edge_index_global=edge_index_global,
            edge_attr_global=edge_attr_global_t0,
            context=context,
        )

        # Compute mu for p(zs | zt).
        mu_x, mu_bond = self.compute_x_pred(
            net_out=eps_t0,
            edge_out=eps_edge_t0,
            z_t=z_0,
            edge_t=edge_attr_global_t0,
            gamma_t=gamma_0,
            data_batch=data_batch,
            source=source,
        )

        xh = self.sample_normal(
            mu=mu_x, sigma=sigma_x, batch=data_batch, fix_noise=fix_noise
        )
        edge_oh = self.sample_symmetric_edge_normal(
            mu_bond, sigma_x[source], edge_index_global, sample_size
        )

        x = xh[:, : self.n_dims]
        h_cat = (
            z_0[:, self.n_dims : -1] if self.include_charges else z_0[:, self.n_dims :]
        )
        h_int = z_0[:, -1:] if self.include_charges else torch.zeros(0).to(z_0.device)

        x, h_cat, h_int, dense_edge_oh = self.unnormalize(x, h_cat, h_int, edge_oh)

        h_cat = F.one_hot(torch.argmax(h_cat, dim=1), num_classes=self.num_classes)
        h_edge = F.one_hot(torch.argmax(dense_edge_oh, dim=1), num_classes=5)
        h_int = torch.round(h_int).long()
        h = {"integer": h_int, "categorical": h_cat, "bonds": h_edge}
        return x, h

    def sample_normal(self, mu, sigma, batch, fix_noise=False):
        """Samples from a Normal distribution."""
        eps = self.sample_combined_position_feature_noise(
            mu.size(0), batch, fix_noise=fix_noise
        )
        return mu + sigma * eps

    def sample_symmetric_edge_normal(
        self, mu_edge, sigma, edge_index_global, sample_size, fix_noise=False
    ):
        noise_edges = torch.randn(sample_size, device=mu_edge.device)
        noise_edges = 0.5 * (noise_edges + noise_edges.permute(1, 0, 2))
        noise_edges = self.dense_to_sparse(noise_edges, edge_index_global)
        return mu_edge + sigma * noise_edges

    def log_pxh_given_z0_without_constants(
        self,
        h,
        edge_attr_global,
        z_t,
        eps,
        net_out,
        eps_edge,
        edge_out,
        batch,
        source,
        gamma,
        epsilon=1e-10,
    ):
        # Discrete properties are predicted directly from z_t.
        z_h_cat = (
            z_t[:, self.n_dims : -1] if self.include_charges else z_t[:, self.n_dims :]
        )
        z_h_int = z_t[:, -1:] if self.include_charges else torch.zeros(0).to(z_t.device)

        # Take only part over x.
        eps_x = eps[:, : self.n_dims]
        net_x = net_out[:, : self.n_dims]

        # Compute sigma_0 and rescale to the integer scale of the data.
        sigma_0 = self.sigma(gamma)[batch]
        sigma_0_cat = sigma_0 * self.norm_values[1]
        sigma_0_int = sigma_0 * self.norm_values[2]

        # Computes the error for the distribution N(x | 1 / alpha_0 z_0 + sigma_0/alpha_0 eps_0, sigma_0 / alpha_0),
        # the weighting in the epsilon parametrization is exactly '1'.
        loss_dict = self.compute_error(
            net_x,
            edge_out,
            eps_x,
            eps_edge,
            batch,
            source,
        )
        log_p_x_given_z_without_constants = -0.5 * (
            loss_dict["coords_loss"]
            + loss_dict["atoms_loss"]
            + loss_dict["charges_loss"]
            + loss_dict["bonds_loss"]
        )

        # Compute delta indicator masks.
        h_integer = torch.round(
            h["integer"] * self.norm_values[2] + self.norm_biases[2]
        ).long()
        onehot = h["categorical"] * self.norm_values[1] + self.norm_biases[1]
        edge_onehot = edge_attr_global * self.norm_values[3] + self.norm_biases[3]

        estimated_h_integer = z_h_int * self.norm_values[2] + self.norm_biases[2]
        estimated_h_cat = z_h_cat * self.norm_values[1] + self.norm_biases[1]
        estimated_edge_cat = edge_out * self.norm_values[3] + self.norm_biases[3]
        assert h_integer.size() == estimated_h_integer.size()

        h_integer_centered = h_integer - estimated_h_integer

        if len(h_integer_centered) == 0:
            log_ph_integer = torch.zeros(len(batch.unique())).to(batch.device)
        else:
            # Compute integral from -0.5 to 0.5 of the normal distribution
            # N(mean=h_integer_centered, stdev=sigma_0_int)
            log_ph_integer = torch.log(
                cdf_standard_gaussian((h_integer_centered + 0.5) / sigma_0_int)
                - cdf_standard_gaussian((h_integer_centered - 0.5) / sigma_0_int)
                + epsilon
            )
            log_ph_integer = torch.sum(scatter_add(log_ph_integer, batch, dim=0), dim=1)

        # Centered h_cat around 1, since onehot encoded.
        centered_h_cat = estimated_h_cat - 1
        centered_edge_cat = estimated_edge_cat - 1

        # Compute integrals from 0.5 to 1.5 of the normal distribution
        # N(mean=z_h_cat, stdev=sigma_0_cat)
        log_ph_cat_proportional = torch.log(
            cdf_standard_gaussian((centered_h_cat + 0.5) / sigma_0_cat)
            - cdf_standard_gaussian((centered_h_cat - 0.5) / sigma_0_cat)
            + epsilon
        )
        log_pedge_cat_proportional = torch.log(
            cdf_standard_gaussian((centered_edge_cat + 0.5) / sigma_0_cat[source])
            - cdf_standard_gaussian((centered_edge_cat - 0.5) / sigma_0_cat[source])
            + epsilon
        )
        # Normalize the distribution over the categories.
        log_Z = torch.logsumexp(log_ph_cat_proportional, dim=1, keepdim=True)
        log_probabilities = log_ph_cat_proportional - log_Z
        log_Z_e = torch.logsumexp(log_pedge_cat_proportional, dim=1, keepdim=True)
        log_probabilities_e = log_pedge_cat_proportional - log_Z_e

        # Select the log_prob of the current category usign the onehot
        # representation.
        log_ph_cat = torch.sum(
            scatter_add(log_probabilities * onehot, batch, dim=0), dim=1
        )

        log_pedge_cat = scatter_mean(
            log_probabilities_e * edge_onehot, index=batch[source], dim=0
        )
        log_pedge_cat = torch.sum(log_pedge_cat, dim=1)
        # Combine categorical and integer log-probabilities.
        log_p_h_given_z = log_ph_integer + log_ph_cat + log_pedge_cat

        # Combine log probabilities for x and h.
        log_p_xh_given_z = log_p_x_given_z_without_constants + log_p_h_given_z

        return log_p_xh_given_z

    def compute_loss(
        self,
        x,
        h,
        data_batch,
        edge_index_global,
        dense_edge_oh,
        context=None,
        t0_always=False,
    ):
        source = edge_index_global[0]
        # This part is about whether to include loss term 0 always.
        if t0_always:
            # loss_term_0 will be computed separately.
            # estimator = loss_0 + loss_t,  where t ~ U({1, ..., T})
            lowest_t = 1
        else:
            # estimator = loss_t,           where t ~ U({0, ..., T})
            lowest_t = 0

        t_int = torch.randint(
            low=lowest_t,
            high=self.hparams["num_diffusion_timesteps"] + 1,
            size=(len(data_batch.unique()), 1),
            dtype=torch.long,
            device=x.device,
        )
        s_int = t_int - 1
        t_is_zero = (t_int == 0).float()  # Important to compute log p(x | z0).

        # Normalize t to [0, 1]. Note that the negative
        # step of s will never be used, since then p(x | z0) is computed.
        s = s_int / self.T
        t = t_int / self.T

        gamma_s = self.gamma(s)
        gamma_t = self.gamma(t)
        alpha_t = self.alpha(gamma_t)[data_batch]
        sigma_t = self.sigma(gamma_t)[data_batch]

        xh = torch.cat([x, h["categorical"], h["integer"]], dim=1)

        # Coordinate and atom types noise
        noise_xh = self.sample_combined_position_feature_noise(
            n_nodes=x.size(0), batch=data_batch
        )
        z_t = alpha_t * xh + sigma_t * noise_xh
        assert_mean_zero(z_t[:, : self.n_dims], data_batch)

        # Edge noise
        noise_edge_oh = self.sample_symmetric_edge_noise(dense_edge_oh=dense_edge_oh)
        noise_edge_oh_global = self.dense_to_sparse(noise_edge_oh, edge_index_global)
        edge_attr_global = self.dense_to_sparse(dense_edge_oh, edge_index_global)
        edge_attr_global_t = (
            alpha_t[source] * edge_attr_global + sigma_t[source] * noise_edge_oh_global
        )
        edge_index_local = radius_graph(
            x=z_t[:, : self.n_dims],
            r=self.hparams["cutoff_upper"],
            batch=data_batch,
            max_num_neighbors=self.hparams["max_num_neighbors"],
        )

        # Neural net prediction.
        eps_t, eps_edge_t = self.network(
            z_t,
            t,
            data_batch,
            edge_index_local,
            edge_index_global,
            edge_attr_global=edge_attr_global_t,
            context=context,
        )
        # Compute the error.
        loss_dict = self.compute_error(
            eps_t,
            eps_edge_t,
            noise_xh,
            noise_edge_oh_global,
            data_batch,
            source,
        )
        error = (
            loss_dict["coords_loss"]
            + loss_dict["atoms_loss"]
            + loss_dict["charges_loss"]
            + loss_dict["bonds_loss"]
        )

        if self.training and self.loss_type == "l2":
            SNR_weight = torch.ones_like(error)
        else:
            # Compute weighting with SNR: (SNR(s-t) - 1) for epsilon parametrization.
            SNR_weight = (self.SNR(gamma_s - gamma_t) - 1).squeeze(1)
        assert error.size() == SNR_weight.size()
        loss_t_larger_than_zero = 0.5 * SNR_weight * error

        # The _constants_ depending on sigma_0 from the
        # cross entropy term E_q(z0 | x) [log p(x | z0)].
        neg_log_constants = -self.log_constants_p_x_given_z0(data_batch)

        # Reset constants during training with l2 loss.
        if self.training and self.loss_type == "l2":
            neg_log_constants = torch.zeros_like(neg_log_constants)

        # The KL between q(z1 | x) and p(z1) = Normal(0, 1). Should be close to zero.
        # kl_prior = torch.zeros(
        #     0, device=xh.device
        # )  # self.kl_prior(xh, eps_edge, data_batch, source)

        # Combining the terms
        if t0_always:
            loss_t = loss_t_larger_than_zero
            num_terms = self.T  # Since t=0 is not included here.
            estimator_loss_terms = num_terms * loss_t

            # Compute noise values for t = 0.
            t_zeros = torch.zeros_like(s)
            gamma_0 = self.gamma(t_zeros)
            alpha_0 = self.alpha(gamma_0)[data_batch]
            sigma_0 = self.sigma(gamma_0)[data_batch]

            # Sample z_0 given x, h for timestep t, from q(z_t | x, h)
            noise_xh_0 = self.sample_combined_position_feature_noise(
                n_nodes=x.size(0), batch=data_batch
            )
            z_0 = alpha_0 * xh + sigma_0 * noise_xh_0

            # Sample edge noise
            noise_edge_oh_0 = self.sample_symmetric_edge_noise(
                dense_edge_oh=dense_edge_oh
            )
            noise_edge_oh_global_0 = self.dense_to_sparse(
                noise_edge_oh_0, edge_index_global
            )
            edge_attr_global_t0 = (
                alpha_0[source] * edge_attr_global
                + sigma_0[source] * noise_edge_oh_global_0
            )

            eps_t0, eps_edge_t0 = self.network(
                z_0,
                t_zeros,
                data_batch,
                edge_index_local,
                edge_index_global,
                edge_attr_global=edge_attr_global_t0,
                context=context,
            )
            loss_term_0 = -self.log_pxh_given_z0_without_constants(
                h=h,
                edge_attr_global=edge_attr_global,
                z_t=z_0,
                eps=noise_xh_0,
                net_out=eps_t0,
                eps_edge=noise_edge_oh_global_0,
                edge_out=eps_edge_t0,
                gamma=gamma_0,
                batch=data_batch,
                source=source,
            )

            # assert kl_prior.size() == estimator_loss_terms.size()
            # assert kl_prior.size() == neg_log_constants.size()
            # assert kl_prior.size() == loss_term_0.size()

            loss = estimator_loss_terms + neg_log_constants + loss_term_0  # + kl_prior

        else:
            # Computes the L_0 term (even if gamma_t is not actually gamma_0)
            # and this will later be selected via masking.
            # loss_term_0 = -self.log_pxh_given_z0_without_constants(
            #     h=h,
            #     z_t=z_t,
            #     eps=eps,
            #     net_out=eps_t,
            #     eps_edge=eps_edge,
            #     edge_out=eps_edge_t,
            #     gamma=gamma_t,
            #     batch=data_batch,
            #     source=source,
            # )
            # t_is_not_zero = 1 - t_is_zero

            # loss_t = (
            #     loss_term_0 * t_is_zero.squeeze()
            #     + t_is_not_zero.squeeze() * loss_t_larger_than_zero
            # )
            loss_t = loss_t_larger_than_zero
            # Only upweigh estimator if using the vlb objective.
            if self.training and self.loss_type == "l2":
                estimator_loss_terms = loss_t
            else:
                num_terms = self.T + 1  # Includes t = 0.
                estimator_loss_terms = num_terms * loss_t

            # assert kl_prior.size() == estimator_loss_terms.size()
            # assert kl_prior.size() == neg_log_constants.size()

            loss = estimator_loss_terms + neg_log_constants  # + kl_prior

        assert len(loss.shape) == 1, f"{loss.shape} has more than only batch dim."

        return loss, loss_dict

    def sample_p_zs_given_zt(
        self,
        s,
        t,
        z_t,
        data_batch,
        edge_index_local,
        edge_index_global,
        edge_attr_global_t,
        sample_size,
        context=None,
        fix_noise=False,
    ):
        source = edge_index_global[0]
        source_l = edge_index_local[0]

        """Samples from zs ~ p(zs | zt). Only used during sampling."""
        gamma_s = self.gamma(s)
        gamma_t = self.gamma(t)
        (
            sigma2_t_given_s,
            sigma_t_given_s,
            alpha_t_given_s,
        ) = self.sigma_and_alpha_t_given_s(gamma_t, gamma_s, data_batch)

        sigma_s = self.sigma(gamma_s)[data_batch]
        sigma_t = self.sigma(gamma_t)[data_batch]

        # Neural net prediction.
        eps_t, eps_edge_t = self.network(
            x=z_t,
            t=t,
            data_batch=data_batch,
            edge_index_local=edge_index_local,
            edge_index_global=edge_index_global,
            edge_attr_global=edge_attr_global_t,
            context=context,
        )

        # Compute mu for p(zs | zt).
        assert_mean_zero(z_t[:, : self.n_dims], data_batch)
        assert_mean_zero(eps_t[:, : self.n_dims], data_batch)
        mu = (
            z_t / alpha_t_given_s
            - (sigma2_t_given_s / alpha_t_given_s / sigma_t) * eps_t
        )
        mu_edge = (
            edge_attr_global_t / alpha_t_given_s[source]
            - (sigma2_t_given_s[source] / alpha_t_given_s[source] / sigma_t[source])
            * eps_edge_t
        )

        # Compute sigma for p(zs | zt).
        sigma = sigma_t_given_s * sigma_s / sigma_t

        # Sample zs given the paramters derived from zt.
        zs = self.sample_normal(mu, sigma, data_batch, fix_noise)
        edge_s = self.sample_symmetric_edge_normal(
            mu_edge, sigma[source], edge_index_global, sample_size, fix_noise
        )

        # Project down to avoid numerical runaway of the center of gravity.
        zs = torch.cat(
            [
                remove_mean(zs[:, : self.n_dims], data_batch),
                zs[:, self.n_dims :],
            ],
            dim=1,
        )
        return zs, edge_s

    def sample_symmetric_edge_noise(self, dense_edge_oh):
        noise_edges = torch.randn_like(dense_edge_oh)
        noise_edges = 0.5 * (noise_edges + noise_edges.permute(1, 0, 2))
        return noise_edges

    def dense_to_sparse(self, dense_edge, edge_index):
        perturbed_edge_attr = dense_edge[edge_index[0, :], edge_index[1, :], :]
        return perturbed_edge_attr

    def sample_combined_position_feature_noise(self, n_nodes, batch, fix_noise=False):
        """
        Samples mean-centered normal noise for z_x, and standard normal noise for z_h.
        """
        z_x = sample_center_gravity_zero_gaussian(
            size=(n_nodes, self.n_dims),
            device=batch.device,
            batch=batch,
            fix_noise=fix_noise,
        )
        z_h = sample_gaussian(
            size=(n_nodes, self.num_atom_features),
            device=batch.device,
            batch=batch,
            fix_noise=fix_noise,
        )
        z = torch.cat([z_x, z_h], dim=1)
        return z

    @torch.no_grad()
    def generate_sample(
        self,
        n_nodes_list,
        context=None,
        fix_noise=False,
    ):
        """
        Draw samples from the generative model.
        """
        batch = torch.cat(
            [
                torch.tensor([0 + i] * n, dtype=torch.int64)
                for i, n in enumerate(n_nodes_list)
            ]
        ).to(self.device)

        num_nodes = batch.bincount().sum()

        dense_edge_oh = torch.zeros(
            (num_nodes, num_nodes, 5), dtype=torch.float32, device=self.device
        )
        sample_size = dense_edge_oh.shape
        edge_index_global = (
            torch.eq(batch.unsqueeze(0), batch.unsqueeze(-1)).int().fill_diagonal_(0)
        )
        edge_index_global, _ = dense_to_sparse(edge_index_global)
        edge_index_global = sort_edge_index(edge_index_global, sort_by_row=False)

        if context is not None and context.shape[0] != n_nodes_list.sum():
            context = context[batch]
            assert (
                context.shape[0] == n_nodes_list.sum()
            ), f"Wrong context shape: {context.shape}!"
            assert (
                context.shape[1] == self.num_context_features
            ), f"Wrong context shape: {context.shape}!"
            assert len(context.size()) == 2, f"Wrong context shape: {context.shape}!"

        n_samples = len(batch.unique())
        # Noise is broadcasted over the batch axis, useful for visualizations.
        z = self.sample_combined_position_feature_noise(
            n_nodes_list.sum(), batch, fix_noise=fix_noise
        )
        edge_index_local = radius_graph(
            x=z[:, : self.n_dims],
            r=self.hparams["cutoff_upper"],
            batch=batch,
            max_num_neighbors=self.hparams["max_num_neighbors"],
        )

        noise_edge_oh = self.sample_symmetric_edge_noise(dense_edge_oh=dense_edge_oh)
        edge_attr_global_t = self.dense_to_sparse(noise_edge_oh, edge_index_global)

        assert_mean_zero(z[:, : self.n_dims], batch)

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        for s in reversed(range(0, self.T)):
            s_array = torch.full((n_samples, 1), fill_value=s, device=z.device)
            t_array = s_array + 1
            s_array = s_array / self.T
            t_array = t_array / self.T

            (
                z,
                edge_attr_global_t,
            ) = self.sample_p_zs_given_zt(
                s=s_array,
                t=t_array,
                z_t=z,
                data_batch=batch,
                edge_index_local=edge_index_local,
                edge_index_global=edge_index_global,
                edge_attr_global_t=edge_attr_global_t,
                sample_size=sample_size,
                context=context,
                fix_noise=fix_noise,
            )

            edge_index_local = radius_graph(
                x=z[:, : self.n_dims],
                r=self.hparams["cutoff_upper"],
                batch=batch,
                max_num_neighbors=self.hparams["max_num_neighbors"],
            )

        # Finally sample p(x, h | z_0).
        x, h = self.sample_p_xh_given_z0(
            z_0=z,
            data_batch=batch,
            edge_index_local=edge_index_local,
            edge_index_global=edge_index_global,
            edge_attr_global_t0=edge_attr_global_t,
            sample_size=sample_size,
            context=context,
            fix_noise=fix_noise,
        )

        assert_mean_zero(x, batch)

        max_cog = torch.sum(x, dim=1, keepdim=True).abs().max().item()
        if max_cog > 5e-2:
            print(
                f"Warning cog drift with error {max_cog:.3f}. Projecting "
                f"the positions down."
            )
            x = remove_mean(x, batch)

        num_nodes = batch.bincount().sum()
        size = torch.Size([num_nodes, num_nodes] + list(h["bonds"].size())[1:])
        adj = torch.sparse_coo_tensor(edge_index_global, h["bonds"], size)
        h["bonds"] = adj.to_dense()

        return x, h, batch

    @torch.no_grad()
    def generate_chain(
        self,
        n_nodes,
        context=None,
        keep_frames=100,
        fix_noise=False,
    ):
        """
        Draw samples from the generative model, keep the intermediate states for visualization purposes.
        """

        batch = torch.zeros(n_nodes, dtype=torch.int64).to(self.device)
        n_samples = len(batch.unique())

        num_nodes = batch.bincount().sum()
        dense_edge_oh = torch.zeros(
            (num_nodes, num_nodes, 5), dtype=torch.float32, device=self.device
        )
        sample_size = dense_edge_oh.shape
        edge_index_global = (
            torch.eq(batch.unsqueeze(0), batch.unsqueeze(-1)).int().fill_diagonal_(0)
        )
        edge_index_global, _ = dense_to_sparse(edge_index_global)
        edge_index_global = sort_edge_index(edge_index_global, sort_by_row=False)

        z = self.sample_combined_position_feature_noise(n_nodes, batch)

        if context is not None and context.shape[0] != n_nodes:
            context = context[batch]
            assert (
                context.shape[0] == z.shape[0]
            ), f"Wrong context shape: {context.shape}!"
            assert (
                context.shape[1] == self.num_context_features
            ), f"Wrong context shape: {context.shape}!"
            assert len(context.size()) == 2, f"Wrong context shape: {context.shape}!"

        edge_index_local = radius_graph(
            x=z[:, : self.n_dims],
            r=self.hparams["cutoff_upper"],
            batch=batch,
            max_num_neighbors=self.hparams["max_num_neighbors"],
        )
        noise_edge_oh = self.sample_symmetric_edge_noise(dense_edge_oh=dense_edge_oh)
        edge_attr_global_t = self.dense_to_sparse(noise_edge_oh, edge_index_global)

        if keep_frames is None:
            keep_frames = self.T
        else:
            assert keep_frames <= self.T
        chain = torch.zeros((keep_frames,) + z.size(), device=z.device)
        chain_bonds = torch.zeros((keep_frames, n_nodes, n_nodes, 5), device=z.device)

        assert_mean_zero(z[:, : self.n_dims], batch)
        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        for s in reversed(range(0, self.T)):
            s_array = torch.full((n_samples, 1), fill_value=s, device=z.device)
            t_array = s_array + 1
            s_array = s_array / self.T
            t_array = t_array / self.T

            z, edge_attr_global_t = self.sample_p_zs_given_zt(
                s=s_array,
                t=t_array,
                z_t=z,
                data_batch=batch,
                edge_index_local=edge_index_local,
                edge_index_global=edge_index_global,
                edge_attr_global_t=edge_attr_global_t,
                sample_size=sample_size,
                context=context,
                fix_noise=fix_noise,
            )

            assert_mean_zero(z[:, : self.n_dims], batch)

            # Write to chain tensor.
            write_index = (s * keep_frames) // self.T
            chain[write_index] = self.unnormalize_z(z, edge_attr_global_t)[0]

            edges = self.unnormalize_z(z, edge_attr_global_t)[1]
            size = torch.Size([n_nodes, n_nodes] + list(edges.size())[1:])
            adj = torch.sparse_coo_tensor(edge_index_global, edges, size)
            edges = adj.to_dense()
            chain_bonds[write_index] = edges

        # Finally sample p(x, h | z_0).
        x, h = self.sample_p_xh_given_z0(
            z_0=z,
            data_batch=batch,
            edge_index_local=edge_index_local,
            edge_index_global=edge_index_global,
            edge_attr_global_t0=edge_attr_global_t,
            sample_size=sample_size,
            context=context,
            fix_noise=fix_noise,
        )

        assert_mean_zero(x[:, : self.n_dims], batch)

        xh = torch.cat([x, h["categorical"], h["integer"]], dim=1)
        chain[0] = xh  # Overwrite last frame with the resulting x and h.

        size = torch.Size([n_nodes, n_nodes] + list(h["bonds"].size())[1:])
        adj = torch.sparse_coo_tensor(edge_index_global, h["bonds"], size)
        h["bonds"] = adj.to_dense()
        chain_bonds[0] = h["bonds"]

        return chain, chain_bonds

    def sample(
        self,
        nodesxsample=torch.tensor([10]),
        fix_noise=False,
        context=None,
    ):
        max_n_nodes = self.dataset_info[
            "max_n_nodes"
        ]  # this is the maximum node_size in the dataset

        assert int(torch.max(nodesxsample)) <= max_n_nodes

        if self.hparams["num_context_features"] > 0:
            if context is None:
                context = self.prop_dist.sample_batch(nodesxsample).to(self.device)
        else:
            context = None

        x, h, batch = self.generate_sample(
            nodesxsample, context=context, fix_noise=fix_noise
        )

        assert_mean_zero(x, batch)

        one_hot = h["categorical"]
        charges = h["integer"]
        bonds = h["bonds"]

        return one_hot, charges, bonds, x, batch

    def sample_chain(self, n_tries, context=None):
        n_samples = 1
        if (
            self.hparams["dataset"] == "qm9"
            or self.hparams["dataset"] == "qm9_2half"
            or self.hparams["dataset"] == "qm9_1half"
        ):
            n_nodes = 19
        elif self.hparams["dataset"] == "drugs":
            n_nodes = 44
        elif self.hparams["dataset"] == "aqm" or self.hparams["dataset"] == "uspto":
            n_nodes = 50
        else:
            raise ValueError()

        # TODO FIX: This conditioning just zeros.
        if self.hparams["num_context_features"] > 0:
            if context is None:
                context = self.prop_dist.sample(n_nodes).unsqueeze(0).to(self.device)
        else:
            context = None

        one_hot, charges, x = None, None, None
        for i in range(n_tries):
            chain, chain_bonds = self.generate_chain(n_nodes, context, keep_frames=100)
            chain = reverse_tensor(chain)
            chain_bonds = reverse_tensor(chain_bonds)

            # Repeat last frame to see final sample better.
            chain = torch.cat([chain, chain[-1:].repeat(10, 1, 1)], dim=0)
            chain_bonds = torch.cat(
                [chain_bonds, chain_bonds[-1:].repeat(10, 1, 1, 1)], dim=0
            )
            x = chain[-1:, :, 0:3]
            one_hot = chain[-1:, :, 3:-1]
            atom_types = torch.argmax(one_hot, dim=2)
            bonds = chain[-1:, :, :, :]
            bond_types = torch.argmax(bonds, dim=3)
            bond_types = bond_types.squeeze(0).cpu().detach().numpy()
            atom_type = atom_types.squeeze(0).cpu().detach().numpy()

            x_squeeze = x.squeeze(0).cpu().detach().numpy()
            q = torch.round(chain[-1:, :, -1:]).long()

            molecule = Molecule(
                atom_types=atom_type,
                positions=x_squeeze,
                charges=q,
                bond_types=bond_types,
                dataset_info=self.dataset_info,
            )
            mol_stable = check_stability_given_bonds(molecule, self.dataset_info)[0]

            # Prepare entire chain.
            charges = (
                chain.shape[-1] - len(self.dataset_info["atom_decoder"]) - 3
            ) == 1
            x = chain[:, :, :3]
            one_hot = chain[:, :, 3:] if not charges else chain[:, :, 3:-1]
            charges = (
                torch.round(chain[:, :, -1:]).long()
                if charges
                else torch.zeros(one_hot.size(0), device=one_hot.device).long()
            )
            chain = PlaceHolder(
                atom_types=one_hot.argmax(2),
                charges=charges,
                bond_types=chain_bonds.argmax(2),
                pos=x,
            )
            if mol_stable:
                print("Found stable molecule to visualize :)")
                break
            elif i == n_tries - 1:
                print("Did not find stable molecule, showing last sample.")

        return molecule, chain, one_hot, charges, bonds, x

    def sample_sweep_conditional(
        self,
        n_nodes=50,
        n_frames=10,
        interpolate_prop=None,
        interpolate_range=None,
    ):
        nodesxsample = torch.tensor([n_nodes] * n_frames)

        context_list = []

        if interpolate_prop is not None:
            if interpolate_range is None:
                context = self.prop_dist.sample(n_nodes).unsqueeze(0)
                idx = self.hparams["properties_list"].index(interpolate_prop)
                interpolate_context = context[:, idx]

                min_val, max_val = self.prop_dist.distributions[interpolate_prop][
                    n_nodes
                ]["params"]
                mean, mad = (
                    self.prop_dist.normalizer[interpolate_prop]["mean"],
                    self.prop_dist.normalizer[interpolate_prop]["mad"],
                )

                min_val = (min_val - mean) / (mad)
                max_val = (max_val - mean) / (mad)

                interpolate_context_up = torch.tensor(
                    np.linspace(interpolate_context, max_val, n_frames // 2)
                )
                interpolate_context_down = torch.tensor(
                    np.linspace(min_val, interpolate_context, n_frames // 2)
                )
                interpolate_context = torch.cat(
                    [interpolate_context_down, interpolate_context_up], dim=0
                )
            else:
                interpolate_min = torch.tensor(
                    [min(interpolate_range)], dtype=torch.float32
                )
                interpolate_max = torch.tensor(
                    [max(interpolate_range)], dtype=torch.float32
                )
                interpolate_context = torch.tensor(
                    np.linspace(interpolate_min, interpolate_max, n_frames)
                )
            context_list.append(interpolate_context)

            for key in self.prop_dist.distributions:
                if key != interpolate_prop:
                    idx = self.hparams["properties_list"].index(key)
                    constant_val = context[:, idx]
                    context_row = constant_val.repeat_interleave(n_frames).unsqueeze(1)
                    context_list.append(context_row)

        else:
            for key in self.prop_dist.distributions:
                min_val, max_val = self.prop_dist.distributions[key][n_nodes]["params"]
                mean, mad = (
                    self.prop_dist.normalizer[key]["mean"],
                    self.prop_dist.normalizer[key]["mad"],
                )
                min_val = (min_val - mean) / (mad)
                max_val = (max_val - mean) / (mad)
                context_row = torch.tensor(
                    np.linspace(min_val, max_val, n_frames)
                ).unsqueeze(1)
                context_list.append(context_row)

        context = torch.cat(context_list, dim=1).float().to(self.device)

        one_hot, charges, bonds, x, batch = self.sample(
            nodesxsample=nodesxsample,
            context=context,
            fix_noise=True,
        )
        return one_hot, charges, bonds, x, batch, context

    def analyze_and_save(
        self,
        n_samples=1000,
        batch_size=100,
        wandb=False,
        path=None,
        test_run=False,
    ):
        if not test_run:
            print(f"Analyzing molecule stability at epoch {self.current_epoch}...")
        else:
            print(f"Analyzing molecule stability for final evaluation...")

        if path is None:
            path = os.path.join(self.hparams["log_dir"], "rdkit_eval.csv")
            pickle_path = os.path.join(self.hparams["log_dir"], "midi_log.pickle")
        else:
            path = os.path.join(path, "rdkit_eval.csv")
            pickle_path = os.path.join(path, "midi_log.pickle")
        batch_size = min(batch_size, n_samples)

        assert n_samples % batch_size == 0

        molecule_list = []
        for _ in range(int(n_samples / batch_size)):
            nodesxsample = self.nodes_dist.sample(batch_size)
            one_hot, charges, bonds, x, batch = self.sample(nodesxsample=nodesxsample)

            molecule_list.extend(
                batch_to_list(
                    one_hot=one_hot.argmax(-1),
                    positions=x,
                    charges=charges,
                    bonds=bonds.argmax(-1),
                    batch=batch,
                    dataset_info=self.dataset_info,
                )
            )

        stability_dict, rdkit_tuple, midi_log = self.val_sampling_metrics(
            molecule_list,
            dataset_info=self.dataset_info,
            bonds_given=True,
            charges_given=self.hparams["include_charges"],
        )
        if rdkit_tuple is not None:
            print(
                f"\n'Molecule stability': {stability_dict['mol_stable']}, 'Atom stability': {stability_dict['atm_stable']}"
            )
            print(
                f"\n'Validity': {rdkit_tuple[0][0]}, 'Uniqueness': {rdkit_tuple[0][1]}, 'Novelty': {rdkit_tuple[0][2]}\n"
            )
        write_to_csv_full(stability_dict, rdkit_tuple, midi_log, path)
        save_log(midi_log, pickle_path)

    def save_and_sample_chain(self, path, id_from=0):
        molecule, chain, one_hot, charges, x = self.sample_chain(n_tries=1)

        vis.save_xyz_file(
            path,
            one_hot,
            charges,
            x,
            self.dataset_info,
            id_from,
            name="chain",
        )
        visualize_chain(
            os.path.join(path, "bonds_chain"),
            chain,
            atom_decoder=self.dataset_infos["atom_decoder"],
        )
        _ = visualize_mol(os.path.join(path, "bonds"), molecule)

        return one_hot, charges, x

    def sample_different_sizes_and_save(
        self,
        path,
        n_samples=5,
        batch_size=100,
    ):
        mols = []
        batch_size = min(batch_size, n_samples)
        for counter in range(int(n_samples / batch_size)):
            nodesxsample = self.nodes_dist.sample(batch_size)
            one_hot, charges, bonds, x, batch = self.sample(
                nodesxsample=nodesxsample,
            )
            mols.extend(
                batch_to_list(
                    one_hot=one_hot.argmax(-1),
                    positions=x,
                    charges=charges,
                    bonds=bonds.argmax(-1),
                    batch=batch,
                    dataset_info=self.dataset_info,
                )
            )
            vis.save_xyz_file(
                path,
                one_hot,
                charges,
                x,
                self.dataset_info,
                batch_size * counter,
                batch=batch,
                name="molecule",
            )
        _ = visualize(os.path.join(path, "bonds"), mols, num_molecules_to_visualize=-1)

    def save_and_sample_conditional(
        self,
        path,
        n_frames=100,
        n_nodes=50,
        interpolate_prop=None,
        id_from=0,
        n_samples=None,
        use_openbabel=False,
        max_iter=None,
    ):
        one_hot, charges, bonds, x, batch, context = self.sample_sweep_conditional(
            n_frames=n_frames,
            n_nodes=n_nodes,
            interpolate_prop=interpolate_prop,
            n_samples=n_samples,
            use_openbabel=use_openbabel,
            max_iter=max_iter,
        )
        vis.save_xyz_file(
            path,
            one_hot,
            charges,
            x,
            self.dataset_info,
            id_from,
            batch=batch,
            name="conditional",
        )
        molecules = batch_to_list(
            one_hot=one_hot.argmax(-1),
            positions=x,
            charges=charges,
            bonds=bonds.argmax(-1),
            batch=batch,
            dataset_info=self.dataset_info,
        )
        _ = visualize(
            os.path.join(path, "bonds"), molecules, num_molecules_to_visualize=-1
        )
        with open(os.path.join(path, "context.pickle"), "wb") as f:
            pickle.dump(context.cpu(), f)

        return one_hot, charges, x


if __name__ == "__main__":
    from qm9.data import QM9DataModule
    from qm9.hparams import add_arguments

    parser = ArgumentParser()
    parser = add_arguments(parser)
    hparams = parser.parse_args()

    if not os.path.exists(hparams.log_dir):
        os.makedirs(hparams.log_dir)

    if not os.path.isdir(hparams.log_dir + f"/run{hparams.id}/"):
        print("Creating directory")
        os.mkdir(hparams.log_dir + f"/run{hparams.id}/")
    print(f"Starting Run {hparams.id}")

    ema_callback = ExponentialMovingAverage(decay=hparams.ema_decay)
    checkpoint_callback = ModelCheckpoint(
        dirpath=hparams.log_dir + f"/run{hparams.id}/",
        save_top_k=1,
        monitor="val/coords_loss",
        save_last=True,
    )
    lr_logger = LearningRateMonitor()
    tb_logger = TensorBoardLogger(
        hparams.log_dir + f"/run{hparams.id}/", default_hp_metric=False
    )

    print(f"Loading {hparams.dataset} Datamodule.")
    datamodule = QM9DataModule(hparams)
    datamodule.prepare_data()
    datamodule.setup("fit")

    dataset_statistics = QM9Infos(datamodule.dataset)

    smiles = datamodule.dataset.smiles
    train_idx = [int(i) for i in datamodule.idx_train]
    train_smiles = [smi for i, smi in enumerate(smiles) if i in train_idx]
    dataset_info = get_dataset_info(hparams.dataset, hparams.remove_hs)

    properties_norm = None
    if len(hparams.properties_list) > 0:
        properties_norm = datamodule.compute_mean_mad(hparams.properties_list)

    dataloader = datamodule.get_dataloader(datamodule.train_dataset, "val")
    nodes_dist, prop_dist = get_distributions(hparams, dataset_info, dataloader)
    if prop_dist is not None:
        prop_dist.set_normalizer(properties_norm)

    model = Trainer(
        hparams=hparams.__dict__,
        dataset_info=dataset_info,
        dataset_statistics=dataset_statistics,
        train_smiles=train_smiles,
        nodes_dist=nodes_dist,
        prop_dist=prop_dist,
        properties_norm=properties_norm,
    )

    strategy = (
        pl.strategies.DDPStrategy(find_unused_parameters=False)
        if hparams.gpus > 1
        else None
    )

    trainer = pl.Trainer(
        accelerator="gpu" if hparams.gpus else "cpu",
        devices=hparams.gpus if hparams.gpus else None,
        strategy=strategy,
        logger=tb_logger,
        enable_checkpointing=True,
        accumulate_grad_batches=hparams.accum_batch,
        val_check_interval=hparams.eval_freq,
        gradient_clip_val=hparams.grad_clip_val,
        callbacks=[
            ema_callback,
            lr_logger,
            checkpoint_callback,
            TQDMProgressBar(refresh_rate=5),
            ModelSummary(max_depth=2),
        ],
        precision=hparams.precision,
        num_sanity_val_steps=2,
        max_epochs=hparams.num_epochs,
        detect_anomaly=hparams.detect_anomaly,
        resume_from_checkpoint=None
        if hparams.load_model is None
        else hparams.load_model,
    )

    pl.seed_everything(seed=0, workers=hparams.gpus > 1)

    trainer.fit(
        model=model,
        datamodule=datamodule,
        # ckpt_path=hparams.load_ckpt if hparams.load_ckpt != "" else None,
    )
