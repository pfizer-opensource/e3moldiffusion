import logging
import os
from datetime import datetime
from typing import List, Tuple, Optional

import pandas as pd
import pytorch_lightning as pl
import numpy as np

import torch
import torch.nn.functional as F
from e3moldiffusion.coordsatomsbonds import (
    DenoisingEdgeNetwork,
    LatentEncoderNetwork,
    SoftMaxAttentionAggregation,
)
from e3moldiffusion.modules import DenseLayer, GatedEquivBlock
from e3moldiffusion.molfeat import get_bond_feature_dims
from experiments.diffusion.categorical import CategoricalDiffusionKernel
from experiments.diffusion.continuous import DiscreteDDPM
from experiments.losses import DiffusionLoss
from experiments.molecule_utils import Molecule
from experiments.sampling.analyze import analyze_stability_for_molecules
from experiments.utils import coalesce_edges, get_list_of_edge_adjs, zero_mean
from experiments.diffusion.utils import initialize_edge_attrs_reverse
from torch import Tensor, nn
from torch_geometric.data import Batch
from torch_geometric.nn import radius_graph
from torch_geometric.utils import dense_to_sparse, sort_edge_index
from tqdm import tqdm

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
        dataset_info: dict,
        smiles_list: list,
        prop_dist=None,
        prop_norm=None,
    ):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.i = 0
        self.mol_valid_cond = 0.25
        self.mol_valid_gen= 0.25

        self.dataset_info = dataset_info

        atom_types_distribution = dataset_info.atom_types.float()
        bond_types_distribution = dataset_info.edge_types.float()
        charge_types_distribution = dataset_info.charges_marginals.float()

        self.register_buffer("atoms_prior", atom_types_distribution.clone())
        self.register_buffer("bonds_prior", bond_types_distribution.clone())
        self.register_buffer("charges_prior", charge_types_distribution.clone())

        self.hparams.num_atom_types = dataset_info.input_dims.X
        self.num_charge_classes = dataset_info.input_dims.C
        self.num_atom_types = self.hparams.num_atom_types
        self.num_atom_features = self.num_atom_types + self.num_charge_classes
        self.num_bond_classes = 5

        self.remove_hs = hparams.get("remove_hs")
        if self.remove_hs:
            print("Model without modelling explicit hydrogens")

        self.smiles_list = smiles_list

        self.dataset_info = dataset_info

        empirical_num_nodes = dataset_info.n_nodes
        self.register_buffer(name="empirical_num_nodes", tensor=empirical_num_nodes)

        self.model = DenoisingEdgeNetwork(
            hn_dim=(hparams["sdim"], hparams["vdim"]),
            latent_dim=hparams["latent_dim"],
            num_layers=hparams["num_layers"],
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
        )

        self.encoder = LatentEncoderNetwork(
            num_atom_features=self.num_atom_types,
            num_bond_types=self.num_bond_classes,
            edge_dim=hparams["edim_latent"],
            cutoff_local=hparams["cutoff_local"],
            hn_dim=(hparams["sdim_latent"], hparams["vdim_latent"]),
            num_layers=hparams["num_layers_latent"],
            vector_aggr=hparams["vector_aggr"],
        )
        self.latent_lin = GatedEquivBlock(
            in_dims=(hparams["sdim_latent"], hparams["vdim_latent"]),
            out_dims=(hparams["latent_dim"], None),
        )
        self.graph_pooling = SoftMaxAttentionAggregation(dim=hparams["latent_dim"])

        self.max_nodes = dataset_info.max_n_nodes
        
        m = 2 if hparams["latentmodel"] == "vae" else 1
        
        self.mu_logvar_z = DenseLayer(hparams["latent_dim"], m * hparams["latent_dim"])
        self.node_z = DenseLayer(hparams["latent_dim"], self.max_nodes)
        
        self.latentloss = PriorLatentLoss(kind=hparams.get("latentmodel"))
        self.latentmodel = get_latent_model(hparams)
        self.latentcache = LatentCache()
        
        self.sde_pos = DiscreteDDPM(
            beta_min=hparams["beta_min"],
            beta_max=hparams["beta_max"],
            N=hparams["timesteps"],
            scaled_reverse_posterior_sigma=True,
            schedule=self.hparams.noise_scheduler,
            nu=2.5,
            enforce_zero_terminal_snr=False,
            param="data"
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
        )
        self.cat_bonds = CategoricalDiffusionKernel(
            terminal_distribution=bond_types_distribution,
            alphas=self.sde_bonds.alphas.clone(),
        )
        self.cat_charges = CategoricalDiffusionKernel(
            terminal_distribution=charge_types_distribution,
            alphas=self.sde_atom_charge.alphas.clone(),
        )

        self.diffusion_loss = DiffusionLoss(
            modalities=["coords", "atoms", "charges", "bonds"],
            param=["data"] * 4
        )

    def training_step(self, batch, batch_idx):
        return self.step_fnc(batch=batch, batch_idx=batch_idx, stage="train")

    def validation_step(self, batch, batch_idx):
        return self.step_fnc(batch=batch, batch_idx=batch_idx, stage="val")

    def my_on_validation_epoch_end(self, conditional):
        final_res = self.run_evaluation(
                    step=self.i,
                    dataset_info=self.dataset_info,
                    ngraphs=1000,
                    bs=self.hparams.inference_batch_size,
                    verbose=True,
                    inner_verbose=False,
                    conditional=conditional,
                    eta_ddim=1.0,
                    ddpm=True,
                    every_k_step=1,
                    device="cuda" if self.hparams.gpus > 1 else "cpu",
                )
        
        if conditional:
            kind = "c"
        else:
            kind = "g"
            
        self.log(
            name=f"val/{kind}_validity",
            value=final_res.validity[0],
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            name=f"val/{kind}_uniqueness",
            value=final_res.uniqueness[0],
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            name=f"val/{kind}_novelty",
            value=final_res.novelty[0],
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            name=f"val/{kind}_mol_stable",
            value=final_res.mol_stable[0],
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            name=f"val/{kind}_atm_stable",
            value=final_res.atm_stable[0],
            on_epoch=True,
            sync_dist=True,
        )
        
    def on_validation_epoch_end(self):
        if (self.current_epoch + 1) % self.hparams.test_interval == 0:
            if self.local_rank == 0:
                print(f"Running reconstruction in epoch {self.current_epoch + 1}")
                print(f"Running generation in epoch {self.current_epoch + 1}")
            self.my_on_validation_epoch_end(conditional=True)
            self.latentcache.empty_cache()
            self.my_on_validation_epoch_end(conditional=False)
            self.i += 1
            
    def _log(
        self,
        loss,
        coords_loss,
        atoms_loss,
        charges_loss,
        bonds_loss,
        prior_loss,
        num_node_loss,
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
            num_node_loss,
            on_step=True,
            batch_size=batch_size,
            prog_bar=(stage == "train"),
            sync_dist=self.hparams.gpus > 1 and stage == "val",
        )
        
    def step_fnc(self, batch, batch_idx, stage: str):
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
            
        out_dict = self(batch=batch, t=t)

        true_data = {
            "coords": out_dict["coords_true"],
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
            batch=batch.batch,
            bond_aggregation_index=out_dict["bond_aggregation_index"],
            weights=weights,
        )

        final_loss = (
            self.hparams.lc_coords * loss["coords"]
            + self.hparams.lc_atoms * loss["atoms"]
            + self.hparams.lc_bonds * loss["bonds"]
            + self.hparams.lc_charges * loss["charges"]
        )

        prior_loss = self.latentloss(inputdict=out_dict.get("latent"))
        num_nodes_loss = F.cross_entropy(out_dict["nodes"]["num_nodes_pred"], out_dict["nodes"]["num_nodes_true"])
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
         
        # save latents in case validation step for generation
        if stage == "val":
            z = out_dict["latent"]["z_true"]
            zs = z.detach().cpu()
            num_nodes = batch.batch.bincount().cpu()
            for z, n in zip(zs.split(1, dim=0), num_nodes.split(1, dim=0), ):
                self.latentcache.update_cache(z, n.item())
                
        return final_loss

    def forward(self, batch: Batch, t: Tensor):
        atom_types: Tensor = batch.x
        pos: Tensor = batch.pos
        charges: Tensor = batch.charges
        data_batch: Tensor = batch.batch
        bond_edge_index = batch.edge_index
        bond_edge_attr = batch.edge_attr
        bs = int(data_batch.max()) + 1

        temb = t.float() / self.hparams.timesteps
        temb = temb.clamp(min=self.hparams.eps_min)
        temb = temb.unsqueeze(dim=1)

        bond_edge_index, bond_edge_attr = sort_edge_index(
            edge_index=bond_edge_index, edge_attr=bond_edge_attr, sort_by_row=False
        )

        atom_types = F.one_hot(
            atom_types.squeeze().long(), num_classes=self.num_atom_types
        ).float()

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
            x=atom_types,
            pos=pos,
            edge_index_local=edge_index_local,
            edge_attr_local=edge_attr_local,
            batch=data_batch,
        )
        latent_out, _ = self.latent_lin(x=(latent_out["s"], latent_out["v"]))
        z = self.graph_pooling(latent_out, data_batch, dim=0, dim_size=bs)
        z = self.mu_logvar_z(z)
        
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
            
        latentdict =  {"z_true": z, "z_pred": zpred, "mu": mu, "logvar": logvar, "w": w, "delta_log_pw": delta_log_pw}  
        
        pred_num_nodes = self.node_z(z)
        true_num_nodes = batch.batch.bincount()
        
        
        # denoising model that parameterizes p(x_t-1 | x_t, z)
        
        if not hasattr(batch, "fc_edge_index"):
            edge_index_global = (
                torch.eq(batch.batch.unsqueeze(0), batch.batch.unsqueeze(-1))
                .int()
                .fill_diagonal_(0)
            )
            edge_index_global, _ = dense_to_sparse(edge_index_global)
            edge_index_global = sort_edge_index(edge_index_global, sort_by_row=False)
        else:
            edge_index_global = batch.fc_edge_index

        edge_index_global, edge_attr_global = coalesce_edges(
            edge_index=edge_index_global,
            bond_edge_index=bond_edge_index,
            bond_edge_attr=bond_edge_attr,
            n=pos.size(0),
        )

        edge_index_global, edge_attr_global = sort_edge_index(
            edge_index=edge_index_global, edge_attr=edge_attr_global, sort_by_row=False
        )

        j, i = edge_index_global
        mask = j < i
        mask_i = i[mask]
        mask_j = j[mask]
        edge_attr_triu = edge_attr_global[mask]
        edge_attr_triu_ohe = F.one_hot(
            edge_attr_triu, num_classes=self.num_bond_classes
        ).float()
        t_edge = t[data_batch[mask_i]]
        probs = self.cat_bonds.marginal_prob(edge_attr_triu_ohe, t=t_edge)
        edges_t_given_0 = probs.multinomial(
            1,
        ).squeeze()
        j = torch.concat([mask_j, mask_i])
        i = torch.concat([mask_i, mask_j])
        edge_index_global_perturbed = torch.stack([j, i], dim=0)
        edge_attr_global_perturbed = torch.concat(
            [edges_t_given_0, edges_t_given_0], dim=0
        )
        edge_index_global_perturbed, edge_attr_global_perturbed = sort_edge_index(
            edge_index=edge_index_global_perturbed,
            edge_attr=edge_attr_global_perturbed,
            sort_by_row=False,
        )

        edge_attr_global_perturbed = F.one_hot(
            edge_attr_global_perturbed, num_classes=self.num_bond_classes
        ).float()

        # Coords: point cloud in R^3
        # sample noise for coords and recenter
        noise_coords_true = torch.randn_like(pos)
        noise_coords_true = zero_mean(
            noise_coords_true, batch=data_batch, dim_size=bs, dim=0
        )
        # center the true point cloud
        pos_centered = zero_mean(pos, data_batch, dim=0, dim_size=bs)
        # get signal and noise coefficients for coords
        mean_coords, std_coords = self.sde_pos.marginal_prob(
            x=pos_centered, t=t[data_batch]
        )
        # perturb coords
        pos_perturbed = mean_coords + std_coords * noise_coords_true

        # one-hot-encode atom types
        probs = self.cat_atoms.marginal_prob(atom_types.float(), t[data_batch])
        atom_types_perturbed = probs.multinomial(
            1,
        ).squeeze()
        atom_types_perturbed = F.one_hot(
            atom_types_perturbed, num_classes=self.num_atom_types
        ).float()

        # one-hot-encode charges
        # offset
        charges = self.dataset_info.one_hot_charges(charges)
        probs = self.cat_charges.marginal_prob(charges.float(), t[data_batch])
        charges_perturbed = probs.multinomial(
            1,
        ).squeeze()
        charges_perturbed = F.one_hot(
            charges_perturbed, num_classes=self.num_charge_classes
        ).float()

        batch_edge_global = data_batch[edge_index_global[0]]

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
            edge_attr_global=edge_attr_global_perturbed,
            batch=data_batch,
            batch_edge_global=batch_edge_global,
        )

        out["coords_perturbed"] = pos_perturbed
        out["atoms_perturbed"] = atom_types_perturbed
        out["charges_perturbed"] = charges_perturbed
        out["bonds_perturbed"] = edge_attr_global_perturbed

        out["coords_true"] = pos_centered
        out["atoms_true"] = atom_types.argmax(dim=-1)
        out["bonds_true"] = edge_attr_global
        out["charges_true"] = charges.argmax(dim=-1)

        out["bond_aggregation_index"] = edge_index_global[1]

        out["latent"] = latentdict
        out["nodes"] = {"num_nodes_pred": pred_num_nodes, "num_nodes_true": true_num_nodes - 1}
        
        return out

    @torch.no_grad()
    def generate_graphs(
        self,
        z: Optional[Tensor],
        batch_num_nodes: Optional[Tensor], 
        num_graphs: int,
        device: torch.device,
        verbose=False,
        save_traj=False,
        ddpm: bool = True,
        eta_ddim: float = 1.0,
        every_k_step: int = 1,
    ):
        (
            pos,
            atom_types,
            charge_types,
            edge_types,
            edge_index_global,
            batch_num_nodes,
            trajs,
        ) = self.reverse_sampling(
            z=z,
            batch_num_nodes=batch_num_nodes,
            num_graphs=num_graphs,
            device=device,
            verbose=verbose,
            save_traj=save_traj,
            ddpm=ddpm,
            every_k_step=every_k_step,
            eta_ddim=eta_ddim
        )

        pos_splits = pos.detach().split(batch_num_nodes.cpu().tolist(), dim=0)

        charge_types_integer = torch.argmax(charge_types, dim=-1)
        # offset back
        charge_types_integer = (
            charge_types_integer - self.dataset_info.charge_offset
        )
        charge_types_integer_split = charge_types_integer.detach().split(
            batch_num_nodes.cpu().tolist(), dim=0
        )
        atom_types_integer = torch.argmax(atom_types, dim=-1)
        atom_types_integer_split = atom_types_integer.detach().split(
            batch_num_nodes.cpu().tolist(), dim=0
        )

        return (
            pos_splits,
            atom_types_integer_split,
            charge_types_integer_split,
            edge_types,
            edge_index_global,
            batch_num_nodes,
            trajs,
        )

    @torch.no_grad()
    def run_evaluation(
        self,
        step: int,
        dataset_info,
        ngraphs: int = 4000,
        bs: int = 500,
        save_dir: str = None,
        return_molecules: bool = False,
        verbose: bool = False,
        inner_verbose=False,
        conditional: bool = False,
        ddpm: bool = True,
        eta_ddim: float = 1.0,
        every_k_step: int = 1,
        run_test_eval: bool = False,
        device: str = "cpu",
    ):
        b = ngraphs // bs
        l = [bs] * b
        if sum(l) != ngraphs:
            l.append(ngraphs - sum(l))
        assert sum(l) == ngraphs

        molecule_list = []
        start = datetime.now()
        if verbose:
            if self.local_rank == 0:
                print(f"Creating {ngraphs} graphs in {l} batches")
        for _, num_graphs in enumerate(l):
            
            if conditional:
                z, batch_num_nodes = self.latentcache.get_cached_latents(n=num_graphs)
                z = z.to(self.empirical_num_nodes.device)
                batch_num_nodes = batch_num_nodes.to(self.empirical_num_nodes.device)
            else:
                z = batch_num_nodes = None
                
            (
                pos_splits,
                atom_types_integer_split,
                charge_types_integer_split,
                edge_types,
                edge_index_global,
                batch_num_nodes,
                _,
            ) = self.generate_graphs(
                z=z, batch_num_nodes=batch_num_nodes,
                num_graphs=num_graphs,
                verbose=inner_verbose,
                device=self.empirical_num_nodes.device,
                save_traj=False,
                ddpm=ddpm,
                eta_ddim=eta_ddim,
                every_k_step=every_k_step,
            )
            
            n = batch_num_nodes.sum().item()
            edge_attrs_dense = torch.zeros(
                size=(n, n, 5), dtype=edge_types.dtype, device=edge_types.device
            )
            edge_attrs_dense[
                edge_index_global[0, :], edge_index_global[1, :], :
            ] = edge_types
            edge_attrs_dense = edge_attrs_dense.argmax(-1)
            edge_attrs_splits = get_list_of_edge_adjs(edge_attrs_dense, batch_num_nodes)

            for positions, atom_types, charges, edges in zip(
                pos_splits,
                atom_types_integer_split,
                charge_types_integer_split,
                edge_attrs_splits,
            ):
                molecule = Molecule(
                    atom_types=atom_types,
                    positions=positions,
                    dataset_info=dataset_info,
                    charges=charges,
                    bond_types=edges,
                )
                molecule_list.append(molecule)

        (
            stability_dict,
            validity_dict,
            statistics_dict,
            all_generated_smiles,
            stable_molecules,
        ) = analyze_stability_for_molecules(
            molecule_list=molecule_list,
            dataset_info=dataset_info,
            smiles_train=self.smiles_list,
            local_rank=self.local_rank,
            return_molecules=return_molecules,
            device=device,
        )
        if conditional:
            if self.mol_valid_cond < validity_dict["validity"] and not run_test_eval:
                self.mol_valid_cond = validity_dict["validity"]
                save_path = os.path.join(self.hparams.save_dir, f"best_mol_valid_conditional.ckpt")
                self.trainer.save_checkpoint(save_path)
        else:
            if self.mol_valid_gen < validity_dict["validity"] and not run_test_eval:
                self.mol_valid_gen = validity_dict["validity"]
                save_path = os.path.join(self.hparams.save_dir, f"best_mol_valid_generative.ckpt")
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
        total_res["ngraphs"] = str(ngraphs)
        
        m = "c" if conditional else "g"
        try:
            if save_dir is None:
                save_dir = os.path.join(
                    self.hparams.save_dir,
                    "run" + str(self.hparams.id),
                    f"evaluation_{m}.csv",
                )
                print(f"Saving evaluation csv file to {save_dir}")
            else:
                save_dir = os.path.join(save_dir, f"evaluation_{m}.csv")
            if self.local_rank == 0:
                with open(save_dir, "a") as f:
                    total_res.to_csv(f, header=True)
        except Exception as e:
            print(e)
            pass

        if return_molecules:
            return total_res, all_generated_smiles, stable_molecules
        else:
            return total_res
            
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
                    sigma_sq_ratio = self.sde_pos.get_sigma_pos_sq_ratio(s_int=s, t_int=t)
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
                        + sqrt_alphas_cumprod_prev
                        * one_m_alphas
                        * z_pred
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
        z: Tensor,
        num_graphs: int,
        device: torch.device,
        verbose: bool = False,
        save_traj: bool = False,
        batch_num_nodes=None, # for reconstruction/conditional
        ddpm: bool = True,
        eta_ddim: float = 1.0,
        every_k_step: int = 1,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, List]:
        
        bs = num_graphs
        
        if z is None: # unconditional sampling aka no reconstruction
            assert batch_num_nodes is None
            z = self.sample_prior_z(bs, device)
            batch_num_nodes = self.node_z(z).argmax(-1) + 1
            batch_num_nodes = batch_num_nodes.detach().long()
        else:
            assert batch_num_nodes is not None
            
        batch = torch.arange(num_graphs,
                             device=device).repeat_interleave(batch_num_nodes,
                                                              dim=0)
        N = len(batch)
        
        # initialiaze the 0-mean point cloud from N(0, I)
        pos = torch.randn(N, 3, device=device, dtype=torch.get_default_dtype())
        pos = zero_mean(pos, batch=batch, dim_size=bs, dim=0)

        n = len(pos)

        # initialize the atom-types
        atom_types = torch.multinomial(
            self.atoms_prior, num_samples=n, replacement=True
        )
        atom_types = F.one_hot(atom_types, self.num_atom_types).float()

        charge_types = torch.multinomial(
            self.charges_prior, num_samples=n, replacement=True
        )
        charge_types = F.one_hot(charge_types, self.num_charge_classes).float()
        
        edge_index_local = None
        edge_index_global = (
            torch.eq(batch.unsqueeze(0), batch.unsqueeze(-1)).int().fill_diagonal_(0)
        )
        edge_index_global, _ = dense_to_sparse(edge_index_global)
        edge_index_global = sort_edge_index(edge_index_global, sort_by_row=False)
        # edge types for FC graph
        (
            edge_attr_global,
            edge_index_global,
            mask,
            mask_i,
        ) = initialize_edge_attrs_reverse(
            edge_index_global, n, self.bonds_prior, self.num_bond_classes, device
        )
        
        batch_edge_global = batch[edge_index_global[0]]

        pos_traj = []
        atom_type_traj = []
        charge_type_traj = []
        edge_type_traj = []

        if self.hparams.continuous_param == "data":
            chain = range(0, self.hparams.timesteps)
        elif self.hparams.continuous_param == "noise":
            chain = range(0, self.hparams.timesteps - 1)

        chain = chain[::every_k_step]
        iterator = (
            tqdm(reversed(chain), total=len(chain)) if verbose else reversed(chain)
        )
        for timestep in iterator:
            s = torch.full(
                size=(bs,), fill_value=timestep, dtype=torch.long, device=pos.device
            )
            t = s + 1 
            temb = t / self.hparams.timesteps
            temb = temb.unsqueeze(dim=1)
            node_feats_in = torch.cat([atom_types, charge_types], dim=-1)
            out = self.model(
                x=node_feats_in,
                t=temb,
                z=z,
                pos=pos,
                edge_index_local=edge_index_local,
                edge_index_global=edge_index_global,
                edge_attr_global=edge_attr_global,
                batch=batch,
                batch_edge_global=batch_edge_global,
            )

            coords_pred = out["coords_pred"].squeeze()
            if ddpm:
                if self.hparams.noise_scheduler == "adaptive":
                    # positions
                    pos = self.sde_pos.sample_reverse_adaptive(
                        s, t, pos, coords_pred, batch, cog_proj=True, eta_ddim=eta_ddim
                    )
                else:
                    # positions
                    pos = self.sde_pos.sample_reverse(
                        t, pos, coords_pred, batch, cog_proj=True, eta_ddim=eta_ddim
                    )
            else:
                pos = self.sde_pos.sample_reverse_ddim(
                    t, pos, coords_pred, batch, cog_proj=True, eta_ddim=eta_ddim
                )
                
            
            # rest
            atoms_pred, charges_pred = out["atoms_pred"].split(
                [self.num_atom_types, self.num_charge_classes], dim=-1
            )
            atoms_pred = atoms_pred.softmax(dim=-1)
            # N x a_0
            edges_pred = out["bonds_pred"].softmax(dim=-1)
            # E x b_0
            charges_pred = charges_pred.softmax(dim=-1)

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
            (
                edge_attr_global,
                edge_index_global,
                mask,
                mask_i,
            ) = self.cat_bonds.sample_reverse_edges_categorical(
                edge_attr_global,
                edges_pred,
                t,
                mask,
                mask_i,
                batch=batch,
                edge_index_global=edge_index_global,
                num_classes=self.num_bond_classes,
            )

            if save_traj:
                pos_traj.append(pos.detach())
                atom_type_traj.append(atom_types.detach())
                edge_type_traj.append(edge_attr_global.detach())
                charge_type_traj.append(charge_types.detach())

        return (
            pos,
            atom_types,
            charge_types,
            edge_attr_global,
            edge_index_global,
            batch_num_nodes,
            [pos_traj, atom_type_traj, charge_type_traj, edge_type_traj],
        )
        
    def configure_optimizers(self):
        all_params = (
            list(self.model.parameters())
            + list(self.encoder.parameters())
            + list(self.latent_lin.parameters())
            + list(self.graph_pooling.parameters())
            + list(self.mu_logvar_z.parameters())
            + list(self.node_z.parameters())
        )
        
        if self.hparams.latentmodel in ["diffusion", "nflow"]:
            all_params += list(self.latentmodel.parameters())
                    
        optimizer = torch.optim.AdamW(
            all_params, lr=self.hparams["lr"], amsgrad=True, weight_decay=1e-12
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
            "monitor": "val/coords_loss_epoch",
            "strict": False,
        }
        return [optimizer], [scheduler]