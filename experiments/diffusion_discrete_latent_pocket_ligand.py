"""
Script that encodes the ligand as latent and the diffusion models denoises a noisy ligand, next to the full protein pocket AND the latent ligand embedding.
So diffusion model inputs (noisy-ligand, full-pocket, latent(ligand))
"""

import logging
import os
import pickle
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Batch
from torch_geometric.nn import radius_graph
from torch_geometric.utils import dense_to_sparse, remove_self_loops, sort_edge_index
from torch_scatter import scatter_mean
from torch_sparse import coalesce
from tqdm import tqdm

from e3moldiffusion.coordsatomsbonds import (
    DenoisingEdgeNetwork,
    LatentEncoderNetwork,
    SoftMaxAttentionAggregation,
)
from e3moldiffusion.latent import compute_mmd
from e3moldiffusion.modules import DenseLayer, GatedEquivBlock
from e3moldiffusion.molfeat import get_bond_feature_dims
from experiments.data.abstract_dataset import AbstractDatasetInfos
from experiments.data.distributions import ConditionalDistributionNodes, prepare_context
from experiments.data.utils import (
    write_trajectory_as_xyz,
    write_xyz_file,
    write_xyz_file_from_batch,
)
from experiments.diffusion.categorical import CategoricalDiffusionKernel
from experiments.diffusion.continuous import DiscreteDDPM
from experiments.diffusion.utils import (
    bond_guidance,
    energy_guidance,
    get_joint_edge_attrs,
    initialize_edge_attrs_reverse,
)
from experiments.losses import DiffusionLoss
from experiments.molecule_utils import Molecule
from experiments.sampling.analyze import analyze_stability_for_molecules
from experiments.utils import (
    coalesce_edges,
    concat_ligand_pocket,
    get_inp_molecules,
    get_list_of_edge_adjs,
    get_molecules,
    load_bond_model,
    load_energy_model,
    load_latent_encoder,
    load_model,
    load_model_ligand,
    remove_mean_pocket,
    zero_mean,
)
from experiments.xtb_energy import calculate_xtb_energy

logging.getLogger("lightning").setLevel(logging.WARNING)
logging.getLogger("pytorch_lightning.utilities.rank_zero").addHandler(
    logging.NullHandler()
)
logging.getLogger("pytorch_lightning.accelerators.cuda").addHandler(
    logging.NullHandler()
)

BOND_FEATURE_DIMS = get_bond_feature_dims()[0]

from e3moldiffusion.latent import LatentCache, PriorLatentLoss, get_latent_model


class Trainer(pl.LightningModule):
    def __init__(
        self,
        hparams: dict,
        dataset_info: AbstractDatasetInfos,
        smiles_list: list,
        histogram: Dict,
        prop_dist=None,
        prop_norm=None,
    ):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.i = 0
        self.validity = 0.0
        self.connected_components = 0.0

        self.dataset_info = dataset_info
        self.prop_norm = prop_norm
        self.prop_dist = prop_dist

        atom_types_distribution = dataset_info.atom_types.float()
        bond_types_distribution = dataset_info.edge_types.float()
        charge_types_distribution = dataset_info.charges_marginals.float()

        self.register_buffer("atoms_prior", atom_types_distribution.clone())
        self.register_buffer("bonds_prior", bond_types_distribution.clone())
        self.register_buffer("charges_prior", charge_types_distribution.clone())

        self.hparams.num_atom_types = dataset_info.input_dims.X
        self.num_charge_classes = dataset_info.input_dims.C
        self.remove_hs = hparams.get("remove_hs")
        if self.remove_hs:
            print("Model without modeling explicit hydrogens")

        self.num_atom_types = self.hparams.num_atom_types
        self.num_atom_features = self.num_atom_types + self.num_charge_classes
        self.num_bond_classes = 5

        self.smiles_list = smiles_list
        self.conditional_size_distribution = ConditionalDistributionNodes(histogram)

        empirical_num_nodes = dataset_info.n_nodes
        self.register_buffer(name="empirical_num_nodes", tensor=empirical_num_nodes)

        if self.hparams.load_ckpt_from_pretrained is not None:
            print("Loading from pre-trained model checkpoint...")

            self.model = load_model_ligand(
                self.hparams.load_ckpt_from_pretrained, self.num_atom_features
            )
            # num_params = len(self.model.state_dict())
            # for i, param in enumerate(self.model.parameters()):
            #     if i < num_params // 2:
            #         param.requires_grad = False
        else:
            self.model = DenoisingEdgeNetwork(
                hn_dim=(hparams["sdim"], hparams["vdim"]),
                num_layers=hparams["num_layers"],
                latent_dim=hparams["latent_dim"],
                use_cross_product=hparams["use_cross_product"],
                num_atom_features=self.num_atom_features,
                num_bond_types=self.num_bond_classes,
                edge_dim=hparams["edim"],
                cutoff_local=hparams["cutoff_local"],
                vector_aggr=hparams["vector_aggr"],
                fully_connected=hparams["fully_connected"],
                local_global_model=hparams["local_global_model"],
                recompute_edge_attributes=True,
                recompute_radius_graph=False,
                edge_mp=hparams["edge_mp"],
                context_mapping=hparams["context_mapping"],
                num_context_features=hparams["num_context_features"],
                bond_prediction=hparams["bond_prediction"],
                property_prediction=hparams["property_prediction"],
                coords_param=hparams["continuous_param"],
                use_pos_norm=hparams["use_pos_norm"],
            )

        self.max_nodes = dataset_info.max_n_nodes

        if self.hparams.load_ckpt_from_pretrained is not None:
            (
                self.encoder,
                self.latent_lin,
                self.graph_pooling,
                self.mu_logvar_z,
                self.node_z,
                self.latentmodel,
            ) = load_latent_encoder(
                filepath=self.hparams.load_ckpt_from_pretrained,
                max_n_nodes=self.max_nodes,
            )
        else:
            self.encoder = LatentEncoderNetwork(
                num_atom_features=self.num_atom_types,
                num_bond_types=self.num_bond_classes,
                edge_dim=hparams["edim_latent"],
                cutoff_local=hparams["cutoff_local"],
                hn_dim=(hparams["sdim_latent"], hparams["vdim_latent"]),
                num_layers=hparams["num_layers_latent"],
                vector_aggr=hparams["vector_aggr"],
                intermediate_outs=hparams["intermediate_outs"],
                use_pos_norm=hparams["use_pos_norm"],
                use_out_norm=hparams["use_out_norm"]
            )
            self.latent_lin = GatedEquivBlock(
                in_dims=(hparams["sdim_latent"], hparams["vdim_latent"]),
                out_dims=(hparams["latent_dim"], None),
            )
            self.graph_pooling = SoftMaxAttentionAggregation(dim=hparams["latent_dim"])
            m = 2 if hparams["latentmodel"] == "vae" else 1
            self.mu_logvar_z = DenseLayer(
                hparams["latent_dim"], m * hparams["latent_dim"]
            )
            self.node_z = DenseLayer(hparams["latent_dim"], self.max_nodes)
            self.latentmodel = get_latent_model(hparams)

        self.latentloss = PriorLatentLoss(kind=hparams.get("latentmodel"))

        self.sde_pos = DiscreteDDPM(
            beta_min=hparams["beta_min"],
            beta_max=hparams["beta_max"],
            N=hparams["timesteps"],
            scaled_reverse_posterior_sigma=True,
            schedule=self.hparams.noise_scheduler,
            nu=2.5,
            enforce_zero_terminal_snr=False,
            T=self.hparams.timesteps,
            param=self.hparams.continuous_param,
        )
        self.sde_atom_charge = DiscreteDDPM(
            beta_min=hparams["beta_min"],
            beta_max=hparams["beta_max"],
            N=hparams["timesteps"],
            scaled_reverse_posterior_sigma=True,
            schedule=self.hparams.noise_scheduler,
            nu=1,
            enforce_zero_terminal_snr=False,
        )
        self.sde_bonds = DiscreteDDPM(
            beta_min=hparams["beta_min"],
            beta_max=hparams["beta_max"],
            N=hparams["timesteps"],
            scaled_reverse_posterior_sigma=True,
            schedule=self.hparams.noise_scheduler,
            nu=1.5,
            enforce_zero_terminal_snr=False,
        )

        self.cat_atoms = CategoricalDiffusionKernel(
            terminal_distribution=atom_types_distribution,
            alphas=self.sde_atom_charge.alphas.clone(),
            num_atom_types=self.num_atom_types,
            num_bond_types=self.num_bond_classes,
            num_charge_types=self.num_charge_classes,
        )
        self.cat_bonds = CategoricalDiffusionKernel(
            terminal_distribution=bond_types_distribution,
            alphas=self.sde_bonds.alphas.clone(),
            num_atom_types=self.num_atom_types,
            num_bond_types=self.num_bond_classes,
            num_charge_types=self.num_charge_classes,
        )
        self.cat_charges = CategoricalDiffusionKernel(
            terminal_distribution=charge_types_distribution,
            alphas=self.sde_atom_charge.alphas.clone(),
            num_atom_types=self.num_atom_types,
            num_bond_types=self.num_bond_classes,
            num_charge_types=self.num_charge_classes,
        )

        self.diffusion_loss = DiffusionLoss(
            modalities=["coords", "atoms", "charges", "bonds"],
            param=["data", "data", "data", "data"],
        )

        if self.hparams.bond_model_guidance:
            print("Using bond model guidance...")
            self.bond_model = load_bond_model(
                self.hparams.ckpt_bond_model, dataset_info
            )
            for param in self.bond_model.parameters():
                param.requires_grad = False
            self.bond_model.eval()

    def training_step(self, batch, batch_idx):
        return self.step_fnc(batch=batch, batch_idx=batch_idx, stage="train")

    def validation_step(self, batch, batch_idx):
        return self.step_fnc(batch=batch, batch_idx=batch_idx, stage="val")

    def test_step(self, batch, batch_idx):
        # return self.step_fnc(batch=batch, batch_idx=batch_idx, stage="test")
        pass

    def on_test_epoch_end(self):
        if self.hparams.use_ligand_dataset_sizes:
            print(f"Running test sampling. Ligand sizes are taken from the data.")
        else:
            print(f"Running test sampling. Ligand sizes are sampled.")
        results_dict, generated_smiles, valid_molecules = self.run_evaluation(
            step=0,
            dataset_info=self.dataset_info,
            verbose=True,
            inner_verbose=True,
            eta_ddim=1.0,
            ddpm=True,
            every_k_step=1,
            device="cpu",
            run_test_eval=True,
            use_ligand_dataset_sizes=self.hparams.use_ligand_dataset_sizes,
            save_dir=self.hparams.test_save_dir,
            save_traj=self.hparams.save_traj,
            return_molecules=True,
        )
        atom_decoder = valid_molecules[0].dataset_info.atom_decoder

        energies = []
        forces_norms = []
        if self.hparams.calculate_energy and not self.hparams.remove_hs:
            for i in range(len(valid_molecules)):
                atom_types = [
                    atom_decoder[int(a)] for a in valid_molecules[i].atom_types
                ]
                try:
                    e, f = calculate_xtb_energy(
                        valid_molecules[i].positions, atom_types
                    )
                except:
                    continue
                valid_molecules[i].energy = e
                valid_molecules[i].forces_norm = f
                energies.append(e)
                forces_norms.append(f)

        if self.hparams.save_xyz:
            context = []
            for i in range(len(valid_molecules)):
                types = [atom_decoder[int(a)] for a in valid_molecules[i].atom_types]
                write_xyz_file(
                    valid_molecules[i].positions,
                    types,
                    os.path.join(self.hparams.test_save_dir, f"mol_{i}.xyz"),
                )
                types_joint = [
                    atom_decoder[int(a)]
                    for a in torch.cat(
                        [
                            valid_molecules[i].atom_types,
                            valid_molecules[i].atom_types_pocket,
                        ],
                        dim=0,
                    )
                ]
                write_xyz_file(
                    torch.cat(
                        [
                            valid_molecules[i].positions,
                            valid_molecules[i].positions_pocket,
                        ],
                        dim=0,
                    ),
                    types_joint,
                    os.path.join(self.hparams.test_save_dir, f"ligand_pocket_{i}.xyz"),
                )
                if self.prop_dist is not None:
                    tmp = []
                    for j, key in enumerate(self.hparams.properties_list):
                        mean, mad = (
                            self.prop_dist.normalizer[key]["mean"],
                            self.prop_dist.normalizer[key]["mad"],
                        )
                        prop = valid_molecules[i].context[j] * mad + mean
                        tmp.append(float(prop))
                    context.append(tmp)

        if self.prop_dist is not None and self.hparams.save_xyz:
            with open(
                os.path.join(self.hparams.test_save_dir, "context.pickle"), "wb"
            ) as f:
                pickle.dump(context, f)
        if self.hparams.calculate_energy and not self.hparams.remove_hs:
            with open(
                os.path.join(self.hparams.test_save_dir, "energies.pickle"), "wb"
            ) as f:
                pickle.dump(energies, f)
            with open(
                os.path.join(self.hparams.test_save_dir, "forces_norms.pickle"), "wb"
            ) as f:
                pickle.dump(forces_norms, f)
        with open(
            os.path.join(self.hparams.test_save_dir, "generated_smiles.pickle"), "wb"
        ) as f:
            pickle.dump(generated_smiles, f)
        with open(
            os.path.join(self.hparams.test_save_dir, "stable_molecules.pickle"), "wb"
        ) as f:
            pickle.dump(valid_molecules, f)

    def on_validation_epoch_end(self):
        if (self.current_epoch + 1) % self.hparams.test_interval == 0:
            if self.local_rank == 0:
                print(f"Running evaluation in epoch {self.current_epoch + 1}")
            final_res = self.run_evaluation(
                step=self.i,
                dataset_info=self.dataset_info,
                verbose=True,
                inner_verbose=False,
                eta_ddim=1.0,
                ddpm=True,
                every_k_step=1,
                device="cuda",
                use_ligand_dataset_sizes=True,
            )
            self.i += 1
            self.log(
                name="val/validity",
                value=final_res.validity[0],
                on_epoch=True,
                sync_dist=True,
            )
            self.log(
                name="val/uniqueness",
                value=final_res.uniqueness[0],
                on_epoch=True,
                sync_dist=True,
            )
            self.log(
                name="val/novelty",
                value=final_res.novelty[0],
                on_epoch=True,
                sync_dist=True,
            )

    def _log(
        self,
        loss,
        coords_loss,
        atoms_loss,
        charges_loss,
        bonds_loss,
        prior_loss,
        num_nodes_loss,
        batch_size,
        stage,
    ):
        self.log(
            f"{stage}/loss",
            loss,
            on_step=True,
            batch_size=batch_size,
            prog_bar=False,
            sync_dist=self.hparams.gpus > 1 and stage == "val",
        )

        self.log(
            f"{stage}/coords_loss",
            coords_loss,
            on_step=True,
            batch_size=batch_size,
            prog_bar=(stage == "train"),
            sync_dist=self.hparams.gpus > 1 and stage == "val",
        )

        self.log(
            f"{stage}/atoms_loss",
            atoms_loss,
            on_step=True,
            batch_size=batch_size,
            prog_bar=(stage == "train"),
            sync_dist=self.hparams.gpus > 1 and stage == "val",
        )

        self.log(
            f"{stage}/charges_loss",
            charges_loss,
            on_step=True,
            batch_size=batch_size,
            prog_bar=(stage == "train"),
            sync_dist=self.hparams.gpus > 1 and stage == "val",
        )

        self.log(
            f"{stage}/bonds_loss",
            bonds_loss,
            on_step=True,
            batch_size=batch_size,
            prog_bar=(stage == "train"),
            sync_dist=self.hparams.gpus > 1 and stage == "val",
        )

        self.log(
            f"{stage}/prior_loss",
            prior_loss,
            on_step=True,
            batch_size=batch_size,
            prog_bar=(stage == "train"),
            sync_dist=self.hparams.gpus > 1 and stage == "val",
        )
        self.log(
            f"{stage}/num_nodes_loss",
            num_nodes_loss,
            on_step=True,
            batch_size=batch_size,
            prog_bar=(stage == "train"),
            sync_dist=self.hparams.gpus > 1 and stage == "val",
        )

    def step_fnc(self, batch, batch_idx, stage: str):
        batch.batch = batch.pos_batch
        batch_size = int(batch.batch.max()) + 1

        t = torch.randint(
            low=1,
            high=self.hparams.timesteps + 1,
            size=(batch_size,),
            dtype=torch.long,
            device=batch.x.device,
        )
        if self.hparams.loss_weighting == "snr_s_t":
            weights = self.sde_atom_charge.snr_s_t_weighting(
                s=t - 1, t=t, device=self.device, clamp_min=0.05, clamp_max=1.5
            )
        elif self.hparams.loss_weighting == "snr_t":
            weights = self.sde_atom_charge.snr_t_weighting(
                t=t,
                device=self.device,
                clamp_min=0.05,
                clamp_max=1.5,
            )
        elif self.hparams.loss_weighting == "exp_t":
            weights = self.sde_atom_charge.exp_t_weighting(t=t, device=self.device)
        elif self.hparams.loss_weighting == "exp_t_half":
            weights = self.sde_atom_charge.exp_t_half_weighting(t=t, device=self.device)
        elif self.hparams.loss_weighting == "uniform":
            weights = None

        if self.hparams.context_mapping:
            context = prepare_context(
                self.hparams["properties_list"],
                self.prop_norm,
                batch,
                self.hparams.dataset,
            )
            batch.context = context

        out_dict = self(batch=batch, t=t, latent_gamma=1.0)

        true_data = {
            "coords": out_dict["coords_true"]
            if self.hparams.continuous_param == "data"
            else out_dict["coords_noise_true"],
            "atoms": out_dict["atoms_true"],
            "charges": out_dict["charges_true"],
            "bonds": out_dict["bonds_true"],
        }

        coords_pred = out_dict["coords_pred"]
        atoms_pred = out_dict["atoms_pred"]
        atoms_pred, charges_pred = atoms_pred.split(
            [self.num_atom_types, self.num_charge_classes], dim=-1
        )
        edges_pred = out_dict["bonds_pred"]

        pred_data = {
            "coords": coords_pred,
            "atoms": atoms_pred,
            "charges": charges_pred,
            "bonds": edges_pred,
        }

        loss = self.diffusion_loss(
            true_data=true_data,
            pred_data=pred_data,
            batch=batch.pos_batch,
            bond_aggregation_index=out_dict["bond_aggregation_index"],
            weights=weights,
        )

        prior_loss = self.latentloss(inputdict=out_dict.get("latent"))
        num_nodes_loss = F.cross_entropy(
            out_dict["nodes"]["num_nodes_pred"], out_dict["nodes"]["num_nodes_true"]
        )

        final_loss = (
            self.hparams.lc_coords * loss["coords"]
            + self.hparams.lc_atoms * loss["atoms"]
            + self.hparams.lc_bonds * loss["bonds"]
            + self.hparams.lc_charges * loss["charges"]
        )

        final_loss = final_loss + self.hparams.prior_beta * prior_loss + num_nodes_loss

        if torch.any(final_loss.isnan()):
            final_loss = final_loss[~final_loss.isnan()]
            print(f"Detected NaNs. Terminating training at epoch {self.current_epoch}")
            exit()
        self._log(
            final_loss,
            loss["coords"],
            loss["atoms"],
            loss["charges"],
            loss["bonds"],
            prior_loss,
            num_nodes_loss,
            batch_size,
            stage,
        )

        return final_loss

    def encode_ligand(
        self, pos, atom_types, data_batch, bond_edge_index, bond_edge_attr
    ):
        bs = len(data_batch.unique())
        # latent encoder
        edge_index_local = radius_graph(
            x=pos,
            r=self.hparams.cutoff_local,
            batch=data_batch,
            max_num_neighbors=128,
            flow="source_to_target",
        )
        edge_index_local, edge_attr_local = coalesce_edges(
            edge_index=edge_index_local,
            bond_edge_index=bond_edge_index,
            bond_edge_attr=bond_edge_attr,
            n=pos.size(0),
        )
        edge_attr_local = F.one_hot(
            edge_attr_local, num_classes=self.num_bond_classes
        ).float()

        latent_out = self.encoder(
            x=F.one_hot(atom_types.long(), num_classes=self.num_atom_types).float(),
            pos=pos,
            edge_index_local=edge_index_local,
            edge_attr_local=edge_attr_local,
            batch=data_batch,
        )
        latent_out, _ = self.latent_lin(x=(latent_out["s"], latent_out["v"]))
        z = self.graph_pooling(latent_out, data_batch, dim=0, dim_size=bs)
        z = self.mu_logvar_z(z)
        return z

    def forward(self, batch: Batch, t: Tensor, latent_gamma: float = 1.0):
        atom_types: Tensor = batch.x
        atom_types_pocket: Tensor = batch.x_pocket
        pos: Tensor = batch.pos
        pos_pocket: Tensor = batch.pos_pocket
        charges: Tensor = batch.charges
        data_batch: Tensor = batch.batch
        data_batch_pocket: Tensor = batch.pos_pocket_batch
        bond_edge_index = batch.edge_index
        bond_edge_attr = batch.edge_attr
        context = batch.context if self.hparams.context_mapping else None
        bond_edge_index, bond_edge_attr = sort_edge_index(
            edge_index=bond_edge_index, edge_attr=bond_edge_attr, sort_by_row=False
        )

        # TIME EMBEDDING
        temb = t.float() / self.hparams.timesteps
        temb = temb.clamp(min=self.hparams.eps_min)
        temb = temb.unsqueeze(dim=1)

        z = self.encode_ligand(
            pos, atom_types, data_batch, bond_edge_index, bond_edge_attr
        )
        # latent prior model
        if self.hparams.latentmodel == "diffusion":
            # train the latent score network
            zmean, zstd = self.sde_pos.marginal_prob(z, t)
            zin = zmean + zstd * torch.randn_like(z)
            if self.hparams.latent_detach:
                zin = zin.detach()
            zpred = self.latentmodel.forward(zin, temb)
            mu = logvar = w = delta_log_pw = None
        elif self.hparams.latentmodel == "nflow":
            # train the latent flow network
            if self.hparams.latent_detach:
                zin = z.detach()
            else:
                zin = z
            w, delta_log_pw = self.latentmodel.f(zin)
            mu = logvar = zpred = None
        elif self.hparams.latentmodel == "mmd":
            mu = logvar = zpred = w = delta_log_pw = None
        elif self.hparams.latentmodel == "vae":
            mu, logvar = z.chunk(2, dim=-1)
            z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
            zpred = w = delta_log_pw = None
        latentdict = {
            "z_true": z,
            "z_pred": zpred,
            "mu": mu,
            "logvar": logvar,
            "w": w,
            "delta_log_pw": delta_log_pw,
        }
        pred_num_nodes = self.node_z(z)
        true_num_nodes = batch.batch.bincount()

        pos_centered, pos_centered_pocket = remove_mean_pocket(
            pos, pos_pocket, data_batch, data_batch_pocket
        )

        # SAMPLING
        noise_coords_true, pos_perturbed = self.sde_pos.sample_pos(
            t,
            pos_centered,
            data_batch,
            remove_mean=False,
        )
        atom_types, atom_types_perturbed = self.cat_atoms.sample_categorical(
            t,
            atom_types,
            data_batch,
            self.dataset_info,
            num_classes=self.num_atom_types,
            type="atoms",
        )
        charges, charges_perturbed = self.cat_charges.sample_categorical(
            t,
            charges,
            data_batch,
            self.dataset_info,
            num_classes=self.num_charge_classes,
            type="charges",
        )

        atom_types_pocket = F.one_hot(
            atom_types_pocket.squeeze().long(), num_classes=self.num_atom_types
        ).float()
        charges_pocket = torch.zeros(
            pos_pocket.shape[0], charges_perturbed.shape[1], dtype=torch.float32
        ).to(self.device)

        # EDGES
        # Fully-connected ligand
        edge_index_global_lig = (
            torch.eq(data_batch.unsqueeze(0), data_batch.unsqueeze(-1))
            .int()
            .fill_diagonal_(0)
        )
        edge_index_global_lig, _ = dense_to_sparse(edge_index_global_lig)
        edge_index_global_lig = sort_edge_index(
            edge_index_global_lig, sort_by_row=False
        )
        edge_index_global_lig, edge_attr_global_lig = coalesce_edges(
            edge_index=edge_index_global_lig,
            bond_edge_index=bond_edge_index,
            bond_edge_attr=bond_edge_attr,
            n=data_batch.size(0),
        )
        edge_index_global_lig, edge_attr_global_lig = sort_edge_index(
            edge_index=edge_index_global_lig,
            edge_attr=edge_attr_global_lig,
            sort_by_row=False,
        )
        edge_attr_global_perturbed_lig = (
            self.cat_bonds.sample_edges_categorical(
                t,
                edge_index_global_lig,
                edge_attr_global_lig,
                data_batch,
                return_one_hot=True,
            )
            if not self.hparams.bond_prediction
            else None
        )
        (
            edge_index_global,
            edge_attr_global_perturbed,
            batch_edge_global,
            edge_mask,
            edge_mask_pocket,
        ) = get_joint_edge_attrs(
            pos_perturbed,
            pos_centered_pocket,
            data_batch,
            data_batch_pocket,
            edge_attr_global_perturbed_lig,
            self.num_bond_classes,
            self.device,
        )
        # Concatenate Ligand-Pocket
        (
            pos_perturbed,
            atom_types_perturbed,
            charges_perturbed,
            batch_full,
            pocket_mask,
        ) = concat_ligand_pocket(
            pos_perturbed,
            pos_centered_pocket,
            atom_types_perturbed,
            atom_types_pocket,
            charges_perturbed,
            charges_pocket,
            data_batch,
            data_batch_pocket,
            sorting=False,
        )

        # Concatenate all node features
        atom_feats_in_perturbed = torch.cat(
            [atom_types_perturbed, charges_perturbed], dim=-1
        )

        out = self.model(
            x=atom_feats_in_perturbed,
            z=z,
            t=temb,
            pos=pos_perturbed,
            edge_index_local=None,
            edge_index_global=edge_index_global,
            edge_index_global_lig=edge_index_global_lig,
            edge_attr_global=edge_attr_global_perturbed
            if not self.hparams.bond_prediction
            else None,
            batch=batch_full,
            batch_edge_global=batch_edge_global,
            context=context,
            pocket_mask=pocket_mask.unsqueeze(1),
            edge_mask=edge_mask,
            edge_mask_pocket=edge_mask_pocket,
            batch_lig=data_batch,
            latent_gamma=latent_gamma,
        )

        # Ground truth masking
        out["coords_true"] = pos_centered
        out["coords_noise_true"] = noise_coords_true
        out["atoms_true"] = atom_types.argmax(dim=-1)
        out["bonds_true"] = edge_attr_global_lig
        out["charges_true"] = charges.argmax(dim=-1)

        out["bond_aggregation_index"] = edge_index_global_lig[1]
        out["latent"] = latentdict
        out["nodes"] = {
            "num_nodes_pred": pred_num_nodes,
            "num_nodes_true": true_num_nodes - 1,
        }

        return out

    @torch.no_grad()
    def run_evaluation(
        self,
        step: int,
        dataset_info,
        save_dir: str = None,
        return_molecules: bool = False,
        verbose: bool = False,
        inner_verbose=False,
        save_traj=False,
        ddpm: bool = True,
        eta_ddim: float = 1.0,
        every_k_step: int = 1,
        use_ligand_dataset_sizes: bool = True,
        build_obabel_mol: bool = False,
        run_test_eval: bool = False,
        guidance_scale: float = 1.0e-4,
        use_energy_guidance: bool = False,
        ckpt_energy_model: str = None,
        n_nodes_bias: int = 0,
        device: str = "cpu",
    ):
        """
        Runs the evaluation on the entire validation dataloader. Generates 1 ligand in 1 receptor structure
        """
        energy_model = None
        if use_energy_guidance:
            energy_model = load_energy_model(ckpt_energy_model, self.num_atom_features)
            # for param in self.energy_model.parameters():
            #    param.requires_grad = False
            energy_model.to(self.device)
            energy_model.eval()

        dataloader = (
            self.trainer.datamodule.val_dataloader()
            if not run_test_eval
            else self.trainer.datamodule.test_dataloader()
        )
        molecule_list = []
        start = datetime.now()
        for i, pocket_data in enumerate(dataloader):
            num_graphs = len(pocket_data.batch.bincount())
            if use_ligand_dataset_sizes:
                num_nodes_lig = pocket_data.batch.bincount().to(self.device)
            else:
                num_nodes_lig = self.conditional_size_distribution.sample_conditional(
                    n1=None, n2=pocket_data.pos_pocket_batch.bincount()
                ).to(self.device)
                num_nodes_lig += n_nodes_bias
            molecules = self.reverse_sampling(
                num_graphs=num_graphs,
                num_nodes_lig=num_nodes_lig,
                pocket_data=pocket_data,
                verbose=inner_verbose,
                save_traj=save_traj,
                ddpm=ddpm,
                eta_ddim=eta_ddim,
                every_k_step=every_k_step,
                guidance_scale=guidance_scale,
                energy_model=energy_model,
                save_dir=save_dir,
                build_obabel_mol=build_obabel_mol,
                iteration=i,
                encode_ligand=True,
            )
            molecule_list.extend(molecules)
        (
            stability_dict,
            validity_dict,
            statistics_dict,
            all_generated_smiles,
            stable_molecules,
            valid_molecules,
        ) = analyze_stability_for_molecules(
            molecule_list=molecule_list,
            dataset_info=dataset_info,
            smiles_train=self.smiles_list,
            local_rank=self.local_rank,
            return_molecules=return_molecules,
            remove_hs=self.hparams.remove_hs,
            device=device,
        )

        if not run_test_eval:
            save_cond = (
                self.validity < validity_dict["validity"]
                and self.connected_components <= statistics_dict["connected_components"]
            )
        else:
            save_cond = False
        if save_cond:
            self.validity = validity_dict["validity"]
            self.connected_components = statistics_dict["connected_components"]
            save_path = os.path.join(self.hparams.save_dir, "best_valid.ckpt")
            self.trainer.save_checkpoint(save_path)

        run_time = datetime.now() - start
        if verbose:
            if self.local_rank == 0:
                print(f"Run time={run_time}")
        total_res = dict(stability_dict)
        total_res.update(validity_dict)
        total_res.update(statistics_dict)
        if self.local_rank == 0:
            print(total_res)
        total_res = pd.DataFrame.from_dict([total_res])
        if self.local_rank == 0:
            print(total_res)

        total_res["step"] = str(step)
        total_res["epoch"] = str(self.current_epoch)
        total_res["run_time"] = str(run_time)
        try:
            if save_dir is None:
                save_dir = os.path.join(
                    self.hparams.save_dir,
                    "run" + str(self.hparams.id),
                    "evaluation.csv",
                )
                print(f"Saving evaluation csv file to {save_dir}")
            else:
                save_dir = os.path.join(save_dir, "evaluation.csv")
            if self.local_rank == 0:
                with open(save_dir, "a") as f:
                    total_res.to_csv(f, header=True)
        except Exception as e:
            print(e)
            pass

        if return_molecules:
            return total_res, all_generated_smiles, valid_molecules
        else:
            return total_res

    @torch.no_grad()
    def generate_ligands(
        self,
        pocket_data,
        num_graphs,
        inner_verbose,
        save_traj,
        ddpm,
        eta_ddim,
        relax_mol=False,
        max_relax_iter=200,
        sanitize=True,
        every_k_step=1,
        fix_n_nodes=False,
        vary_n_nodes=None,
        n_nodes_bias=0,
        build_obabel_mol=False,
        save_dir=None,
    ):
        if fix_n_nodes:
            num_nodes_lig = pocket_data.batch.bincount().to(self.device)
            if vary_n_nodes:
                num_nodes_lig += torch.randint(
                    low=0, high=n_nodes_bias, size=num_nodes_lig.size()
                ).to(self.device)
            else:
                num_nodes_lig += n_nodes_bias
        else:
            try:
                pocket_size = pocket_data.pos_pocket_batch.bincount()[0].unsqueeze(0)
                num_nodes_lig = (
                    self.conditional_size_distribution.sample_conditional(
                        n1=None, n2=pocket_size
                    )
                    .repeat(num_graphs)
                    .to(self.device)
                )
            except Exception:
                print(
                    "Could not retrieve ligand size from the conditional size distribution given the pocket size. Taking the ground truth size."
                )
                num_nodes_lig = pocket_data.batch.bincount().to(self.device)
            if vary_n_nodes:
                num_nodes_lig += torch.randint(
                    low=0, high=n_nodes_bias, size=num_nodes_lig.size()
                ).to(self.device)
            else:
                num_nodes_lig += n_nodes_bias

        molecules = self.reverse_sampling(
            num_graphs=num_graphs,
            num_nodes_lig=num_nodes_lig,
            pocket_data=pocket_data,
            verbose=inner_verbose,
            save_traj=save_traj,
            ddpm=ddpm,
            eta_ddim=eta_ddim,
            every_k_step=every_k_step,
            guidance_scale=None,
            energy_model=None,
            relax_mol=relax_mol,
            max_relax_iter=max_relax_iter,
            sanitize=sanitize,
            build_obabel_mol=build_obabel_mol,
            save_dir=save_dir,
            encode_ligand=True,  # note this is hardcoded as true for our use-case on conditioning. (evaluation but also inference)
        )
        return molecules

    def sample_prior_z(self, bs, device):
        z = torch.randn(bs, self.hparams.latent_dim, device=device)

        if self.hparams.latentmodel == "diffusion":
            chain = range(self.hparams.timesteps)
            iterator = reversed(chain)
            for timestep in iterator:
                s = torch.full(
                    size=(bs,), fill_value=timestep, dtype=torch.long, device=device
                )
                t = s + 1
                temb = t / self.hparams.timesteps
                temb = temb.unsqueeze(dim=1)
                z_pred = self.latentmodel.forward(z, temb)
                if self.hparams.noise_scheduler == "adaptive":
                    sigma_sq_ratio = self.sde_pos.get_sigma_pos_sq_ratio(
                        s_int=s, t_int=t
                    )
                    z_t_prefactor = (
                        self.sde_pos.get_alpha_pos_ts(t_int=t, s_int=s) * sigma_sq_ratio
                    ).unsqueeze(-1)
                    x_prefactor = self.sde_pos.get_x_pos_prefactor(
                        s_int=s, t_int=t
                    ).unsqueeze(-1)

                    prefactor1 = self.sde_pos.get_sigma2_bar(t_int=t)
                    prefactor2 = self.sde_pos.get_sigma2_bar(
                        t_int=s
                    ) * self.sde_pos.get_alpha_pos_ts_sq(t_int=t, s_int=s)
                    sigma2_t_s = prefactor1 - prefactor2
                    noise_prefactor_sq = sigma2_t_s * sigma_sq_ratio
                    noise_prefactor = torch.sqrt(noise_prefactor_sq).unsqueeze(-1)

                    mu = z_t_prefactor * z + x_prefactor * z_pred
                    noise = torch.randn_like(z)
                    z = mu + noise_prefactor * noise
                else:
                    rev_sigma = self.sde_pos.reverse_posterior_sigma[t].unsqueeze(-1)
                    sigmast = self.sde_pos.sqrt_1m_alphas_cumprod[t].unsqueeze(-1)
                    sigmas2t = sigmast.pow(2)

                    sqrt_alphas = self.sde_pos.sqrt_alphas[t].unsqueeze(-1)
                    sqrt_1m_alphas_cumprod_prev = torch.sqrt(
                        1.0 - self.sde_pos.alphas_cumprod_prev[t]
                    ).unsqueeze(-1)
                    one_m_alphas_cumprod_prev = sqrt_1m_alphas_cumprod_prev.pow(2)
                    sqrt_alphas_cumprod_prev = torch.sqrt(
                        self.sde_pos.alphas_cumprod_prev[t].unsqueeze(-1)
                    )
                    one_m_alphas = self.sde_pos.discrete_betas[t].unsqueeze(-1)

                    mean = (
                        sqrt_alphas * one_m_alphas_cumprod_prev * z
                        + sqrt_alphas_cumprod_prev * one_m_alphas * z_pred
                    )
                    mean = (1.0 / sigmas2t) * mean
                    std = rev_sigma
                    noise = torch.randn_like(mean)
                    z = mean + std * noise
        elif self.hparams.latentmodel == "nflow":
            z = self.latentmodel.g(z).view(bs, -1)

        return z

    def reverse_sampling(
        self,
        num_graphs: int,
        pocket_data: Tensor,
        num_nodes_lig: int = None,
        verbose: bool = False,
        save_traj: bool = False,
        ddpm: bool = True,
        eta_ddim: float = 1.0,
        every_k_step: int = 1,
        guidance_scale: float = 1.0e-4,
        energy_model=None,
        save_dir: str = None,
        relax_mol=False,
        max_relax_iter=200,
        sanitize=False,
        build_obabel_mol=False,
        iteration: int = 0,
        encode_ligand: bool = False,
        latent_gamma: float = 1.0,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, List]:
        pos_pocket = pocket_data.pos_pocket.to(self.device)
        batch_pocket = pocket_data.pos_pocket_batch.to(self.device)
        x_pocket = pocket_data.x_pocket.to(self.device)

        bs = num_graphs

        if encode_ligand:
            # encode ligand
            atom_types: Tensor = pocket_data.x.to(self.device)
            atom_types_pocket: Tensor = pocket_data.x_pocket.to(self.device)
            pos: Tensor = pocket_data.pos.to(self.device)
            pos_pocket: Tensor = pocket_data.pos_pocket.to(self.device)
            charges: Tensor = pocket_data.charges.to(self.device)
            data_batch: Tensor = pocket_data.batch.to(self.device)
            data_batch_pocket: Tensor = pocket_data.pos_pocket_batch.to(self.device)
            bond_edge_index = pocket_data.edge_index.to(self.device)
            bond_edge_attr = pocket_data.edge_attr.to(self.device)
            context = (
                pocket_data.context.to(self.device)
                if self.hparams.context_mapping
                else None
            )
            bond_edge_index, bond_edge_attr = sort_edge_index(
                edge_index=bond_edge_index, edge_attr=bond_edge_attr, sort_by_row=False
            )
            z = self.encode_ligand(
                pos.to(self.device),
                atom_types.to(self.device),
                data_batch.to(self.device),
                bond_edge_index.to(self.device),
                bond_edge_attr.to(self.device),
            )
            batch_num_nodes = pocket_data.batch.bincount().to(self.device)
        else:
            z = self.sample_prior_z(bs, self.device)
            if (
                self.hparams.latentmodel == "diffusion"
                or self.hparams.latentmodel == "nflow"
            ):
                batch_num_nodes = self.node_z(z).argmax(-1) + 1
                batch_num_nodes = batch_num_nodes.detach().long()
            else:
                batch_num_nodes = pocket_data.batch.bincount().to(self.device)
                # batch_num_nodes = self.conditional_size_distribution.sample_conditional(
                #    n1=None, n2=pocket_data.pos_pocket_batch.bincount()
                # ).to(self.device)

        if num_nodes_lig is None:
            num_nodes_lig = batch_num_nodes

        batch = torch.arange(num_graphs, device=self.device).repeat_interleave(
            num_nodes_lig, dim=0
        )
        bs = int(batch.max()) + 1

        # sample context condition
        context = None
        if self.prop_dist is not None:
            context = self.prop_dist.sample_batch(num_nodes_lig).to(self.device)[batch]

        # initialize the 0-mean point cloud from N(0, I) centered in the pocket
        pocket_cog = scatter_mean(pos_pocket, batch_pocket, dim=0)
        pocket_cog_batch = pocket_cog[batch]
        pos = pocket_cog_batch + torch.randn_like(pocket_cog_batch)
        # # project to COM-free subspace
        pos, pos_pocket = remove_mean_pocket(pos, pos_pocket, batch, batch_pocket)
        n = len(pos)

        # initialize the atom-types
        atom_types = torch.multinomial(
            self.atoms_prior, num_samples=n, replacement=True
        )
        atom_types = F.one_hot(atom_types, self.num_atom_types).float()
        atom_types_pocket = F.one_hot(
            x_pocket.squeeze().long(), num_classes=self.num_atom_types
        ).float()

        charge_types = torch.multinomial(
            self.charges_prior, num_samples=n, replacement=True
        )
        charge_types = F.one_hot(charge_types, self.num_charge_classes).float()
        charges_pocket = torch.zeros(
            pos_pocket.shape[0], charge_types.shape[1], dtype=torch.float32
        ).to(self.device)

        edge_index_local = None
        edge_index_global = (
            torch.eq(batch.unsqueeze(0), batch.unsqueeze(-1)).int().fill_diagonal_(0)
        )
        edge_index_global, _ = dense_to_sparse(edge_index_global)
        edge_index_global = sort_edge_index(edge_index_global, sort_by_row=False)
        if not self.hparams.bond_prediction:
            (
                edge_attr_global_lig,
                edge_index_global_lig,
                mask,
                mask_i,
            ) = initialize_edge_attrs_reverse(
                edge_index_global,
                n,
                self.bonds_prior,
                self.num_bond_classes,
                self.device,
            )
        else:
            edge_attr_global = None

        (
            edge_index_global,
            edge_attr_global,
            batch_edge_global,
            edge_mask,
            edge_mask_pocket,
        ) = get_joint_edge_attrs(
            pos,
            pos_pocket,
            batch,
            batch_pocket,
            edge_attr_global_lig,
            self.num_bond_classes,
            self.device,
        )

        (
            pos_joint,
            atom_types_joint,
            charge_types_joint,
            batch_full,
            pocket_mask,
        ) = concat_ligand_pocket(
            pos,
            pos_pocket,
            atom_types,
            atom_types_pocket,
            charge_types,
            charges_pocket,
            batch,
            batch_pocket,
            sorting=False,
        )

        if self.hparams.continuous_param == "data":
            chain = range(0, self.hparams.timesteps)
        elif self.hparams.continuous_param == "noise":
            chain = range(0, self.hparams.timesteps - 1)

        chain = chain[::every_k_step]

        iterator = (
            tqdm(reversed(chain), total=len(chain)) if verbose else reversed(chain)
        )
        for i, timestep in enumerate(iterator):
            s = torch.full(
                size=(bs,), fill_value=timestep, dtype=torch.long, device=pos.device
            )
            t = s + 1

            temb = t / self.hparams.timesteps
            temb = temb.unsqueeze(dim=1)

            node_feats_in = torch.cat([atom_types_joint, charge_types_joint], dim=-1)
            out = self.model(
                x=node_feats_in,
                t=temb,
                z=z,
                pos=pos_joint,
                edge_index_local=edge_index_local,
                edge_index_global=edge_index_global,
                edge_index_global_lig=edge_index_global_lig,
                edge_attr_global=edge_attr_global,
                batch=batch_full,
                batch_edge_global=batch_edge_global,
                context=context,
                pocket_mask=pocket_mask.unsqueeze(1),
                edge_mask=edge_mask,
                edge_mask_pocket=edge_mask_pocket,
                batch_lig=batch,
                latent_gamma=latent_gamma,
            )

            coords_pred = out["coords_pred"].squeeze()
            atoms_pred, charges_pred = out["atoms_pred"].split(
                [self.num_atom_types, self.num_charge_classes], dim=-1
            )
            atoms_pred = atoms_pred.softmax(dim=-1)
            # N x a_0
            edges_pred = out["bonds_pred"].softmax(dim=-1)
            # E x b_0
            charges_pred = charges_pred.softmax(dim=-1)

            if ddpm:
                if self.hparams.noise_scheduler == "adaptive":
                    # positions
                    pos = self.sde_pos.sample_reverse_adaptive(
                        s, t, pos, coords_pred, batch, cog_proj=False, eta_ddim=eta_ddim
                    )  # here is cog_proj false as it will be downprojected later
                else:
                    # positions
                    pos = self.sde_pos.sample_reverse(
                        t, pos, coords_pred, batch, cog_proj=False, eta_ddim=eta_ddim
                    )  # here is cog_proj false as it will be downprojected later
            else:
                pos = self.sde_pos.sample_reverse_ddim(
                    t, pos, coords_pred, batch, cog_proj=False, eta_ddim=eta_ddim
                )  # here is cog_proj false as it will be downprojected later

            # atoms
            atom_types = self.cat_atoms.sample_reverse_categorical(
                xt=atom_types,
                x0=atoms_pred,
                t=t[batch],
                num_classes=self.num_atom_types,
            )
            # charges
            charge_types = self.cat_charges.sample_reverse_categorical(
                xt=charge_types,
                x0=charges_pred,
                t=t[batch],
                num_classes=self.num_charge_classes,
            )
            # edges
            if not self.hparams.bond_prediction:
                (
                    edge_attr_global_lig,
                    edge_index_global_lig,
                    mask,
                    mask_i,
                ) = self.cat_bonds.sample_reverse_edges_categorical(
                    edge_attr_global_lig,
                    edges_pred,
                    t,
                    mask,
                    mask_i,
                    batch=batch,
                    edge_index_global=edge_index_global_lig,
                    num_classes=self.num_bond_classes,
                )
            else:
                edge_attr_global_lig = edges_pred

            (
                edge_index_global,
                edge_attr_global,
                batch_edge_global,
                edge_mask,
                edge_mask_pocket,
            ) = get_joint_edge_attrs(
                pos,
                pos_pocket,
                batch,
                batch_pocket,
                edge_attr_global_lig,
                self.num_bond_classes,
                self.device,
            )
            # pos, pos_pocket = remove_mean_pocket(pos, pos_pocket, batch, batch_pocket)
            (
                pos_joint,
                atom_types_joint,
                charge_types_joint,
                batch_full,
                pocket_mask,
            ) = concat_ligand_pocket(
                pos,
                pos_pocket,
                atom_types,
                atom_types_pocket,
                charge_types,
                charges_pocket,
                batch,
                batch_pocket,
                sorting=False,
            )

            if save_traj:
                atom_decoder = self.dataset_info.atom_decoder
                write_xyz_file_from_batch(
                    pos,
                    atom_types,
                    batch,
                    pos_pocket=pos_pocket,
                    atoms_pocket=atom_types_pocket,
                    batch_pocket=batch_pocket,
                    joint_traj=True,
                    atom_decoder=atom_decoder,
                    path=os.path.join(save_dir, f"iter_{iteration}"),
                    i=i,
                )

        # Move generated molecule back to the original pocket position
        pos += pocket_cog_batch
        pos_pocket += pocket_cog[batch_pocket]

        out_dict = {
            "coords_pred": pos,
            "coords_pocket": pos_pocket,
            "atoms_pred": atom_types,
            "atoms_pocket": atom_types_pocket,
            "charges_pred": charge_types,
            "bonds_pred": edge_attr_global_lig,
            "z": z,
        }
        molecules = get_molecules(
            out_dict,
            batch,
            edge_index_global_lig,
            self.num_atom_types,
            self.num_charge_classes,
            self.dataset_info,
            data_batch_pocket=batch_pocket,
            device=self.device,
            mol_device="cpu",
            context=context,
            relax_mol=relax_mol,
            max_relax_iter=max_relax_iter,
            sanitize=sanitize,
            while_train=False,
            build_obabel_mol=build_obabel_mol,
        )

        if save_traj:
            write_trajectory_as_xyz(
                molecules,
                strict=False,
                joint_traj=True,
                path=os.path.join(save_dir, f"iter_{iteration}"),
            )

        return molecules

    def configure_optimizers(self):
        all_params = (
            list(self.model.parameters())
            + list(self.encoder.parameters())
            + list(self.latent_lin.parameters())
            + list(self.graph_pooling.parameters())
            + list(self.mu_logvar_z.parameters())
            + list(self.node_z.parameters())
        )
        if self.hparams.optimizer == "adam":
            optimizer = torch.optim.AdamW(
                all_params,
                lr=self.hparams["lr"],
                amsgrad=True,
                weight_decay=1.0e-12,
            )
        elif self.hparams.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                all_params,
                lr=self.hparams["lr"],
                momentum=0.9,
                nesterov=True,
            )
        if self.hparams["lr_scheduler"] == "reduce_on_plateau":
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                patience=self.hparams["lr_patience"],
                cooldown=self.hparams["lr_cooldown"],
                factor=self.hparams["lr_factor"],
            )
        elif self.hparams["lr_scheduler"] == "cyclic":
            lr_scheduler = torch.optim.lr_scheduler.CyclicLR(
                optimizer,
                base_lr=self.hparams["lr_min"],
                max_lr=self.hparams["lr"],
                mode="exp_range",
                step_size_up=self.hparams["lr_step_size"],
                cycle_momentum=False,
            )
        elif self.hparams["lr_scheduler"] == "one_cyclic":
            lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.hparams["lr"],
                steps_per_epoch=len(self.trainer.datamodule.train_dataset),
                epochs=self.hparams["num_epochs"],
            )
        elif self.hparams["lr_scheduler"] == "cosine_annealing":
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.hparams["lr_patience"],
                eta_min=self.hparams["lr_min"],
            )
        scheduler = {
            "scheduler": lr_scheduler,
            "interval": "epoch",
            "frequency": self.hparams["lr_frequency"],
            "monitor": self.validity,
            "strict": False,
        }
        return [optimizer], [scheduler]
