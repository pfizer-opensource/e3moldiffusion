import logging
import os
import pickle
import sys
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import RDConfig
from torch import Tensor
from torch_geometric.data import Batch
from torch_geometric.utils import dense_to_sparse, sort_edge_index
from torch_scatter import scatter_mean
from tqdm import tqdm

sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

from e3moldiffusion.coordsatomsbonds import DenoisingEdgeNetwork
from e3moldiffusion.molfeat import get_bond_feature_dims
from experiments.data.abstract_dataset import AbstractDatasetInfos
from experiments.data.distributions import ConditionalDistributionNodes, prepare_context
from experiments.data.utils import (
    get_fc_edge_index_with_offset,
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
    property_classifier_guidance,
    self_guidance,
)
from experiments.losses import DiffusionLoss
from experiments.sampling.analyze import analyze_stability_for_molecules
from experiments.sampling.utils import calculate_sascore
from experiments.utils import (
    coalesce_edges,
    concat_ligand_pocket,
    get_molecules,
    load_bond_model,
    load_energy_model,
    load_model_ligand,
    load_property_model,
    remove_mean_pocket,
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

        # backward compatability:
        if "joint_property_prediction" not in hparams.keys():
            hparams["joint_property_prediction"] = False
        if "atoms_continuous" not in hparams.keys():
            hparams["atoms_continuous"] = False
        if "bonds_continuous" not in hparams.keys():
            hparams["bonds_continuous"] = False
        if "store_intermediate_coords" not in hparams.keys():
            hparams["store_intermediate_coords"] = False
        if "ligand_pocket_distance_loss" not in hparams.keys():
            hparams["ligand_pocket_distance_loss"] = False
        if "ligand_pocket_hidden_distance" not in hparams.keys():
            hparams["ligand_pocket_hidden_distance"] = False

        self.save_hyperparameters(hparams)

        self.i = 0
        self.validity = 0.0
        self.connected_components = 0.0
        self.qed = 0.0

        self.dataset_info = dataset_info
        self.prop_norm = prop_norm
        self.prop_dist = prop_dist

        atom_types_distribution = dataset_info.atom_types.float()
        if self.hparams.num_bond_classes != 5:
            bond_types_distribution = torch.zeros(
                (self.hparams.num_bond_classes,), dtype=torch.float32
            )
            bond_types_distribution[:5] = dataset_info.edge_types.float()
        else:
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
        self.num_bond_classes = self.hparams.num_bond_classes

        self.smiles_list = smiles_list

        self.conditional_size_distribution = ConditionalDistributionNodes(histogram)

        if self.hparams.load_ckpt_from_pretrained is not None:
            print("Loading from pre-trained model checkpoint...")

            self.model = load_model_ligand(
                self.hparams.load_ckpt_from_pretrained,
                self.num_atom_features,
                self.num_bond_classes,
                hparams=self.hparams,
            )
            # num_params = len(self.model.state_dict())
            # for i, param in enumerate(self.model.parameters()):
            #     if i < num_params // 2:
            #         param.requires_grad = False
        else:
            self.model = DenoisingEdgeNetwork(
                hn_dim=(hparams["sdim"], hparams["vdim"]),
                num_layers=hparams["num_layers"],
                latent_dim=None,
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
                coords_param=hparams["continuous_param"],
                use_pos_norm=hparams["use_pos_norm"],
                # ligand_pocket_interaction=False, #hparams["ligand_pocket_interaction"],  # to test old model
                ligand_pocket_interaction=hparams["ligand_pocket_interaction"],
                store_intermediate_coords=hparams["store_intermediate_coords"],
                distance_ligand_pocket=hparams["ligand_pocket_hidden_distance"],
                bond_prediction=hparams["bond_prediction"],
                property_prediction=hparams["property_prediction"],
                joint_property_prediction=hparams["joint_property_prediction"],
            )
            
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

        if not self.hparams.atoms_continuous:
            self.cat_atoms = CategoricalDiffusionKernel(
                terminal_distribution=atom_types_distribution,
                alphas=self.sde_atom_charge.alphas.clone(),
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

        if not self.hparams.bonds_continuous:
            self.cat_bonds = CategoricalDiffusionKernel(
                terminal_distribution=bond_types_distribution,
                alphas=self.sde_bonds.alphas.clone(),
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

        if self.hparams.ligand_pocket_distance_loss:
            self.dist_loss = torch.nn.HuberLoss(reduction="none", delta=1.0)
        else:
            self.dist_loss = None

    def training_step(self, batch, batch_idx):
        return self.step_fnc(batch=batch, batch_idx=batch_idx, stage="train")

    def validation_step(self, batch, batch_idx):
        return self.step_fnc(batch=batch, batch_idx=batch_idx, stage="val")

    def test_step(self, batch, batch_idx):
        # return self.step_fnc(batch=batch, batch_idx=batch_idx, stage="test")
        pass

    def on_test_epoch_end(self):
        if self.hparams.use_ligand_dataset_sizes:
            print("Running test sampling. Ligand sizes are taken from the data.")
        else:
            print("Running test sampling. Ligand sizes are sampled.")
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
            build_obabel_mol=self.hparams.build_obabel_mol,
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
            os.path.join(self.hparams.test_save_dir, "valid_molecules.pickle"), "wb"
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
        sa_loss,
        docking_loss,
        dloss,
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
        
        if sa_loss is not None:
            self.log(
                f"{stage}/sa_loss",
                sa_loss,
                on_step=True,
                batch_size=batch_size,
                prog_bar=(stage == "train"),
                sync_dist=self.hparams.gpus > 1 and stage == "val",
            )
            
        if docking_loss is not None:
            self.log(
                f"{stage}/docking_loss",
                docking_loss,
                on_step=True,
                batch_size=batch_size,
                prog_bar=(stage == "train"),
                sync_dist=self.hparams.gpus > 1 and stage == "val",
            )

        if dloss is not None:
            self.log(
                f"{stage}/d_loss",
                dloss,
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

        molsize_weights = (
            batch.batch.bincount()
            if self.hparams.molsize_loss_weighting and stage == "train"
            else None
        )

        if self.hparams.context_mapping:
            context = prepare_context(
                self.hparams["properties_list"],
                self.prop_norm,
                batch,
                self.hparams.dataset,
            )
            batch.context = context

        out_dict = self(batch=batch, t=t)

        true_data = {
            "coords": out_dict["coords_true"]
            if self.hparams.continuous_param == "data"
            else out_dict["coords_noise_true"],
            "atoms": out_dict["atoms_true"],
            "charges": out_dict["charges_true"],
            "bonds": out_dict["bonds_true"],
            #"properties": out_dict["properties_true"]
            #if False # self.hparams.joint_property_prediction
            #else None,
            "properties": None,
        }

        coords_pred = out_dict["coords_pred"]
        atoms_pred = out_dict["atoms_pred"]
        atoms_pred, charges_pred = atoms_pred.split(
            [self.num_atom_types, self.num_charge_classes], dim=-1
        )
        edges_pred = out_dict["bonds_pred"]
        #prop_pred = (
        #    out_dict["property_pred"].squeeze()
        #    if self.hparams.joint_property_prediction
        #    else None
        #)

        pred_data = {
            "coords": coords_pred,
            "atoms": atoms_pred,
            "charges": charges_pred,
            "bonds": edges_pred,
            # "properties": prop_pred,
            "properties": None,
        }

        loss = self.diffusion_loss(
            true_data=true_data,
            pred_data=pred_data,
            batch=batch.pos_batch,
            bond_aggregation_index=out_dict["bond_aggregation_index"],
            intermediate_coords=self.hparams.store_intermediate_coords
            and self.training,
            weights=weights,
            molsize_weights=molsize_weights,
            regression_property=self.hparams.regression_property,
        )

        final_loss = (
            self.hparams.lc_coords * loss["coords"]
            + self.hparams.lc_atoms * loss["atoms"]
            + self.hparams.lc_bonds * loss["bonds"]
            + self.hparams.lc_charges * loss["charges"]
            # + self.hparams.lc_properties * loss["properties"]
        )
        
        sa_true, docking_true = out_dict["properties_true"]
        sa_pred, docking_pred = out_dict["property_pred"]
        sa_pred, docking_pred = sa_pred.squeeze(dim=1), docking_pred.squeeze(dim=1)
        # sa
        docking_loss = F.mse_loss(input=docking_pred, target=docking_true, reduction="none")
        docking_loss = torch.mean(weights * docking_loss)
        sa_loss = F.binary_cross_entropy_with_logits(input=sa_pred, target=sa_true, reduction="none")
        sa_loss = torch.mean(weights * sa_loss)
        
        final_loss = final_loss + sa_loss + docking_loss
        
        if self.hparams.ligand_pocket_distance_loss:
            coords_pocket = out_dict["distance_loss_data"]["pos_centered_pocket"]
            ligand_i, pocket_j = out_dict["distance_loss_data"]["edge_index_cross"]
            dloss_true = (
                (out_dict["coords_true"][ligand_i] - coords_pocket[pocket_j])
                .pow(2)
                .sum(-1)
                .sqrt()
            )
            dloss_pred = (
                (out_dict["coords_pred"][ligand_i] - coords_pocket[pocket_j])
                .pow(2)
                .sum(-1)
                .sqrt()
            )
            # geometry loss
            dloss = self.dist_loss(dloss_true, dloss_pred).mean()
            if self.hparams.ligand_pocket_hidden_distance:
                d_hidden = out_dict["dist_pred"]
                # latent loss
                dloss1 = self.dist_loss(dloss_true, d_hidden).mean()
                # consistency loss between geometry and latent
                dloss2 = self.dist_loss(dloss_pred, d_hidden).mean()
                dloss = dloss + dloss1 + dloss2
            final_loss = final_loss + 1.0 * dloss
        else:
            dloss = None

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
            sa_loss,
            docking_loss,
            dloss,
            batch_size,
            stage,
        )

        return final_loss

    def forward(self, batch: Batch, t: Tensor):
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

        pocket_noise = torch.randn_like(pos_pocket) * self.hparams.pocket_noise_std
        pos_pocket = pos_pocket + pocket_noise

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

        if not self.hparams.atoms_continuous:
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
        else:
            atom_types = F.one_hot(atom_types, num_classes=self.num_atom_types).float()
            if self.hparams.continuous_param == "noise":
                atom_types = 0.25 * atom_types

            # sample noise for OHEs in {0, 1}^NUM_CLASSES
            noise_atom_types = torch.randn_like(atom_types)
            mean_ohes, std_ohes = self.sde_atom_charge.marginal_prob(
                x=atom_types, t=t[data_batch]
            )
            # perturb OHEs
            atom_types_perturbed = mean_ohes + std_ohes * noise_atom_types

            # Charges
            charges = self.dataset_info.one_hot_charges(charges).float()
            # sample noise for OHEs in {0, 1}^NUM_CLASSES
            noise_charges = torch.randn_like(charges)
            mean_ohes, std_ohes = self.sde_atom_charge.marginal_prob(
                x=charges, t=t[data_batch]
            )
            # perturb OHEs
            charges_perturbed = mean_ohes + std_ohes * noise_charges

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

        if self.hparams.bonds_continuous:
            n = len(pos)
            # create block diagonal matrix
            dense_edge = torch.zeros(n, n, device=self.device, dtype=torch.long)
            # populate entries with integer features
            dense_edge[
                edge_index_global_lig[0, :], edge_index_global_lig[1, :]
            ] = edge_attr_global_lig
            dense_edge_ohe = (
                F.one_hot(dense_edge.view(-1, 1), num_classes=BOND_FEATURE_DIMS + 1)
                .view(n, n, -1)
                .float()
            )

            assert (
                torch.norm(dense_edge_ohe - dense_edge_ohe.permute(1, 0, 2)).item()
                == 0.0
            )

            # create symmetric noise for edge-attributes
            noise_edges = torch.randn_like(dense_edge_ohe)
            noise_edges = 0.5 * (noise_edges + noise_edges.permute(1, 0, 2))
            assert torch.norm(noise_edges - noise_edges.permute(1, 0, 2)).item() == 0.0

            signal = self.sde_bonds.sqrt_alphas_cumprod[t]
            std = self.sde_bonds.sqrt_1m_alphas_cumprod[t]

            signal_b = signal[data_batch].unsqueeze(-1).unsqueeze(-1)
            std_b = std[data_batch].unsqueeze(-1).unsqueeze(-1)
            dense_edge_ohe_perturbed = dense_edge_ohe * signal_b + noise_edges * std_b

            # retrieve as edge-attributes in PyG Format
            edge_attr_global_perturbed_lig = dense_edge_ohe_perturbed[
                edge_index_global_lig[0, :], edge_index_global_lig[1, :], :
            ]
            edge_attr_global_noise = noise_edges[
                edge_index_global_lig[0, :], edge_index_global_lig[1, :], :
            ]
        else:
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
            ca_mask=batch.pocket_ca_mask,
            batch_pocket=batch.pos_pocket_batch,
        )

        # Ground truth masking
        out["coords_true"] = pos_centered
        out["coords_noise_true"] = noise_coords_true
        if self.hparams.atoms_continuous:
            out["atoms_noise_true"] = noise_atom_types
            out["charges_noise_true"] = noise_charges
        out["atoms_true"] = atom_types.argmax(dim=-1)
        out["bonds_true"] = edge_attr_global_lig
        out["charges_true"] = charges.argmax(dim=-1)
        out["bond_aggregation_index"] = edge_index_global_lig[1]
        
        if self.hparams.joint_property_prediction:
            #if self.hparams.regression_property == "sascore":
            label0 = (
                torch.tensor([calculate_sascore(mol) for mol in batch.mol])
                .to(self.device)
                .float()
            )
            #elif self.hparams.regression_property == "docking_score":
            label1 = batch.docking_scores.float()
            out["properties_true"] = (label0, label1)
        if self.hparams.bonds_continuous:
            out["bonds_noise_true"] = edge_attr_global_noise

        if self.hparams.ligand_pocket_distance_loss:
            # Protein Pocket Coords for Distance Loss computation
            # Only select subset based on C-alpha representatives
            data_batch_pocket = data_batch_pocket[batch.pocket_ca_mask]
            # create cross indices between ligand and c-alpha
            adj_cross = (data_batch[:, None] == data_batch_pocket[None, :]).nonzero().T
            out["distance_loss_data"] = {
                "pos_centered_pocket": pos_centered_pocket[batch.pocket_ca_mask],
                "edge_index_cross": adj_cross,
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
        use_ligand_dataset_sizes: bool = False,
        build_obabel_mol: bool = False,
        run_test_eval: bool = False,
        guidance_scale: float = 1.0e-4,
        property_guidance: bool = False,
        ckpt_property_model: str = None,
        n_nodes_bias: int = 0,
        device: str = "cpu",
    ):
        """
        Runs the evaluation on the entire validation dataloader. Generates 1 ligand in 1 receptor structure
        """

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
                property_guidance=property_guidance,
                ckpt_property_model=ckpt_property_model,
                save_dir=save_dir,
                build_obabel_mol=build_obabel_mol,
                iteration=i,
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
        vary_n_nodes=False,
        n_nodes_bias=0,
        property_guidance=None,
        ckpt_property_model=None,
        property_self_guidance=False,
        guidance_scale=None,
        build_obabel_mol=False,
        save_dir=None,
        importance_sampling=False,
        tau=0.1,
        importance_sampling_start=0,
        importance_sampling_end=200,
        every_importance_t=5,
        maximize_score=True,
        with_docking: bool = False,
        tau1: float = 1.0,
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
            property_guidance=property_guidance,
            ckpt_property_model=ckpt_property_model,
            property_self_guidance=property_self_guidance,
            guidance_scale=guidance_scale,
            relax_mol=relax_mol,
            max_relax_iter=max_relax_iter,
            sanitize=sanitize,
            build_obabel_mol=build_obabel_mol,
            save_dir=save_dir,
            importance_sampling=importance_sampling,
            tau=tau,
            importance_sampling_start=importance_sampling_start,
            importance_sampling_end=importance_sampling_end,
            every_importance_t=every_importance_t,
            maximize_score=maximize_score,
            with_docking=with_docking,
            tau1=tau1,
        )
        return molecules


    def docking_guidance(self,
                          node_feats_in,
                            temb,
                            pos,
                            edge_index_local,
                            edge_index_global,
                            edge_attr_global,
                            batch,
                            batch_lig,
                            batch_edge_global,
                            context,
                            batch_num_nodes,
                            edge_index_global_lig: Tensor,
                            edge_attr_global_lig: Tensor,
                            pocket_mask: Tensor,
                            edge_mask: Tensor,
                            ca_mask: Tensor,
                            edge_mask_pocket: Tensor,
                            batch_pocket: Tensor,
                            tau: float = 1e-3,
                            normalize_grad: bool = False,
                          ):
        
        if not pos.requires_grad:
            pos.requires_grad = True
        
        with torch.enable_grad():
            out = self.model(
                    x=node_feats_in,
                    t=temb,
                    pos=pos,
                    edge_index_local=edge_index_local,
                    edge_index_global=edge_index_global,
                    edge_index_global_lig=edge_index_global_lig,
                    edge_attr_global=edge_attr_global,
                    batch=batch,
                    batch_edge_global=batch_edge_global,
                    context=context,
                    pocket_mask=pocket_mask.unsqueeze(1),
                    edge_mask=edge_mask,
                    edge_mask_pocket=edge_mask_pocket,
                    batch_lig=batch_lig,
                    ca_mask=ca_mask,
                    batch_pocket=batch_pocket,
                )
        
        _, docking = out["property_pred"]
        grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(docking)]
        grad_shift = torch.autograd.grad(
            [docking],
            [pos],
            grad_outputs=grad_outputs,
            create_graph=False,
            retain_graph=False,
        )[0]
        
        grad_shift = grad_shift * pocket_mask.float()
        if normalize_grad:
            grad_shift = F.normalize(grad_shift, dim=1, p=2)
        pos = pos.detach()
        pos = pos - tau * grad_shift
        return pos
    
    def importance_sampling(self, 
                             node_feats_in,
                             temb,
                             pos,
                             edge_index_local,
                             edge_index_global,
                             edge_attr_global,
                             batch,
                             batch_lig,
                             batch_edge_global,
                             context,
                             batch_num_nodes,
                             edge_index_global_lig: Tensor,
                             edge_attr_global_lig: Tensor,
                             pocket_mask: Tensor,
                             edge_mask: Tensor,
                             ca_mask: Tensor,
                             edge_mask_pocket: Tensor,
                             batch_pocket: Tensor,
                             tau: float = 1.0,
                             maximize_score: bool = True,
                             with_docking: bool = False,
                             tau1 : float = 1.0,
                             ):
        """
        Idea: 
        The point clouds / graphs have an intermediate predicted synthesizability. 
        Given a set/population of B graphs/point clouds we want to __bias__ the sampling process towards "regions" where the fitness (here the synth.) is maximized.
        Hence we can compute importance weights for each sample i=1,2,...,B and draw a new population with replacement. 
        As the sampling process is stochastic, repeated samples will evolve differently. 
        However we need to think about ways to also include/enforce uniformity such that some samples are not drawn too often. 
        To make it more "uniform", we can use temperature annealing in the softmax
        """
        
        out = self.model(
                x=node_feats_in,
                t=temb,
                pos=pos,
                edge_index_local=edge_index_local,
                edge_index_global=edge_index_global,
                edge_index_global_lig=edge_index_global_lig,
                edge_attr_global=edge_attr_global,
                batch=batch,
                batch_edge_global=batch_edge_global,
                context=context,
                pocket_mask=pocket_mask.unsqueeze(1),
                edge_mask=edge_mask,
                edge_mask_pocket=edge_mask_pocket,
                batch_lig=batch_lig,
                ca_mask=ca_mask,
                batch_pocket=batch_pocket,
            )
        
        pocket_mask = pocket_mask.bool()
        node_feats_in = node_feats_in[pocket_mask]
        pos = pos[pocket_mask]

        # TBD change into tuple for (sa_pred, docking_pred)
        sa, docking = out["property_pred"]
        sa = sa.squeeze(dim=1).sigmoid() # [B,]
        docking = docking.squeeze(dim=1)
        
        if not maximize_score:
            sa = 1.0 - sa
        n = pos.size(0)
        b = len(batch_num_nodes)
        
        weights0 = (sa / tau).softmax(dim=0)
        weights1 = (docking / tau1).softmax(dim=0)
        
        if with_docking:
            weights = (weights0 + weights1) / 0.01
            weights = weights.softmax(dim=0)
        else:
            weights = weights0
            
        select = torch.multinomial(weights, num_samples=len(weights),  replacement=True)        
        select = select.sort()[0]
        ptr = torch.concat([torch.zeros((1,), device=batch_num_nodes.device, dtype=torch.long),
                            batch_num_nodes.cumsum(0)], 
                           dim=0
                           )
        batch_num_nodes_new = batch_num_nodes[select]
        # select 
        batch_new = torch.arange(b, device=pos.device).repeat_interleave(batch_num_nodes_new)
        ## node level
        a, b = node_feats_in.size(1), pos.size(1)
        x = torch.concat([node_feats_in, pos], dim=1)
        x_split = x.split(batch_num_nodes.cpu().numpy().tolist(), dim=0)
        x_select = torch.concat([x_split[i] for i in select.cpu().numpy()], dim=0)
        node_feats_in, pos = x_select.split([a, b], dim=-1)
        
        ## edge level
        edge_slices = [slice(ptr[i-1].item(), ptr[i].item()) for i in range(1, len(ptr))]
        edge_slices_new = [edge_slices[i] for i in select.cpu().numpy()]
        
        # populate the dense edge-tensor
        E_dense = torch.zeros((n, n, edge_attr_global_lig.size(1)),
                              dtype=edge_attr_global_lig.dtype,
                              device=edge_attr_global_lig.device
                              )
        E_dense[edge_index_global_lig[0], edge_index_global_lig[1], :] = edge_attr_global_lig
        
        # select the slices
        E_s = torch.stack([torch.block_diag(*[E_dense[s, s, i] for s in edge_slices_new]) for i in range(E_dense.size(-1))], dim=-1)
        new_ptr = torch.concat([torch.zeros((1,), device=batch_num_nodes_new.device, dtype=torch.long),
                                batch_num_nodes_new.cumsum(0)],
                               dim=0
                               )
        
        new_fc_edge_index = torch.concat([get_fc_edge_index_with_offset(n=batch_num_nodes_new[i].item(),
                                                                        offset=new_ptr[i].item()
                                                                        )
                                          for i in range(len(new_ptr)-1)
                                          ], dim=1
                                         )
        
        new_edge_attr = E_s[new_fc_edge_index[0], new_fc_edge_index[1], :]
        # batch_edge_global = batch_new[new_fc_edge_index[0]]
        # batch_edge_global = None
        
        out =  pos.to(self.device), node_feats_in.to(self.device),\
            new_fc_edge_index.to(self.device), new_edge_attr.to(self.device),\
                batch_new.to(self.device), None, batch_num_nodes_new.to(self.device)
        return out
            
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
        property_guidance: bool = False,
        ckpt_property_model: str = None,
        property_self_guidance: bool = False,
        guidance_scale: float = 1.0e-4,
        save_dir: str = None,
        relax_mol=False,
        max_relax_iter=200,
        sanitize=False,
        build_obabel_mol=False,
        iteration: int = 0,
        importance_sampling: bool = False,
        tau: float = 0.1,
        every_importance_t: int = 5,
        importance_sampling_start: int = 0,
        importance_sampling_end: int = 200,
        maximize_score: bool = True,
        with_docking: bool = False,
        tau1: float = 1.0,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, List]:
        pos_pocket = pocket_data.pos_pocket.to(self.device)
        batch_pocket = pocket_data.pos_pocket_batch.to(self.device)
        x_pocket = pocket_data.x_pocket.to(self.device)

        try:
            ca_mask = pocket_data.ca_mask.to(self.device)
        except:
            ca_mask = None
            
        batch = torch.arange(num_graphs, device=self.device).repeat_interleave(
            num_nodes_lig, dim=0
        )
        bs = int(batch.max()) + 1

        if property_self_guidance or property_guidance:
            t = torch.arange(0, self.hparams.timesteps)
            alphas = self.sde_pos.alphas_cumprod[t]
        if property_guidance and ckpt_property_model is not None:
            property_model = load_property_model(
                ckpt_property_model, self.num_atom_features
            )
            property_model.to(self.device)
            property_model.eval()

        # sample context condition
        context = None
        if self.prop_dist is not None:
            context = self.prop_dist.sample_batch(num_nodes_lig).to(self.device)[batch]

        # initialize the 0-mean point cloud from N(0, I) centered in the pocket
        pocket_cog = scatter_mean(pos_pocket, batch_pocket, dim=0)
        pocket_cog_batch = pocket_cog[batch]
        pos = pocket_cog_batch + torch.randn_like(pocket_cog_batch)
        # pos = pocket_data.pos.to(self.device)
        # batch = pocket_data.batch.to(self.device)

        # # project to COM-free subspace
        pos, pos_pocket = remove_mean_pocket(pos, pos_pocket, batch, batch_pocket)

        n = len(pos)

        if not self.hparams.atoms_continuous:
            # initialize the atom- and charge types
            atom_types = torch.multinomial(
                self.atoms_prior, num_samples=n, replacement=True
            )
            atom_types = F.one_hot(atom_types, self.num_atom_types).float()

            charge_types = torch.multinomial(
                self.charges_prior, num_samples=n, replacement=True
            )
            charge_types = F.one_hot(charge_types, self.num_charge_classes).float()
        else:
            # initialize the atom- and charge types
            atom_types = torch.randn(
                pos.size(0), self.num_atom_types, device=self.device
            )
            charge_types = torch.randn(
                pos.size(0), self.num_charge_classes, device=self.device
            )

        atom_types_pocket = F.one_hot(
            x_pocket.squeeze().long(), num_classes=self.num_atom_types
        ).float()
        charges_pocket = torch.zeros(
            pos_pocket.shape[0], charge_types.shape[1], dtype=torch.float32
        ).to(self.device)

        if self.hparams.bonds_continuous:
            edge_index_local = None
            edge_index_global_lig = (
                torch.eq(batch.unsqueeze(0), batch.unsqueeze(-1))
                .int()
                .fill_diagonal_(0)
            )
            # sample symmetric edge-attributes
            edge_attrs = torch.randn(
                (
                    edge_index_global_lig.size(0),
                    edge_index_global_lig.size(1),
                    self.num_bond_classes,
                ),
                device=self.device,
                dtype=torch.get_default_dtype(),
            )
            # symmetrize
            edge_attrs = 0.5 * (edge_attrs + edge_attrs.permute(1, 0, 2))
            assert torch.norm(edge_attrs - edge_attrs.permute(1, 0, 2)).item() == 0.0
            # get COO format (2, E)
            edge_index_global_lig, _ = dense_to_sparse(edge_index_global_lig)
            edge_index_global_lig = sort_edge_index(
                edge_index_global_lig, sort_by_row=False
            )
            # select in PyG formt (E, self.hparams.num_bond_types)
            edge_attr_global_lig = edge_attrs[
                edge_index_global_lig[0, :], edge_index_global_lig[1, :], :
            ]
            batch_edge_global_lig = batch[edge_index_global_lig[0]]
        else:
            edge_index_local = None
            edge_index_global = (
                torch.eq(batch.unsqueeze(0), batch.unsqueeze(-1))
                .int()
                .fill_diagonal_(0)
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
                ca_mask=ca_mask,
                batch_pocket=batch_pocket,
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
                    if self.hparams.atoms_continuous:
                        atom_types = self.sde_atom_charge.sample_reverse_adaptive(
                            s,
                            t,
                            atom_types,
                            atoms_pred,
                            batch,
                        )
                        charge_types = self.sde_atom_charge.sample_reverse_adaptive(
                            s,
                            t,
                            charge_types,
                            charges_pred,
                            batch,
                        )
                    if self.hparams.bonds_continuous:
                        edge_attr_global_lig = self.sde_bonds.sample_reverse_adaptive(
                            s,
                            t,
                            edge_attr_global_lig,
                            edges_pred,
                            batch_edge_global_lig,
                            edge_attrs=edge_attrs,
                            edge_index_global=edge_index_global_lig,
                        )
                else:
                    # positions
                    pos = self.sde_pos.sample_reverse(
                        t, pos, coords_pred, batch, cog_proj=False, eta_ddim=eta_ddim
                    )  # here is cog_proj false as it will be downprojected later
            else:
                pos = self.sde_pos.sample_reverse_ddim(
                    t, pos, coords_pred, batch, cog_proj=False, eta_ddim=eta_ddim
                )  # here is cog_proj false as it will be downprojected later

            if not self.hparams.atoms_continuous:
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
            if not self.hparams.bonds_continuous:
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
            
            if property_self_guidance:
                signal = 1.0  # alphas[timestep] / (guidance_scale * 10)
                (
                    pos,
                    atom_types,
                    charge_types,
                ) = self_guidance(
                    model=self.model,
                    pos=pos,
                    pos_pocket=pos_pocket,
                    atom_types=atom_types,
                    atom_types_pocket=atom_types_pocket,
                    charge_types=charge_types,
                    charges_pocket=charges_pocket,
                    edge_index_global=edge_index_global,
                    edge_index_global_lig=edge_index_global_lig,
                    edge_attr_global=edge_attr_global,
                    batch=batch,
                    batch_pocket=batch_pocket,
                    batch_full=batch_full,
                    batch_edge_global=batch_edge_global,
                    batch_size=bs,
                    pocket_mask=pocket_mask,
                    edge_mask=edge_mask,
                    edge_mask_pocket=edge_mask_pocket,
                    ca_mask=pocket_data.pocket_ca_mask.to(batch.device),
                    num_atom_types=self.num_atom_types,
                    temb=temb,
                    context=context,
                    signal=signal,
                    guidance_scale=guidance_scale,
                    optimization="maximize",
                )
                
            elif property_guidance:
                signal = alphas[timestep] / (guidance_scale * 10)
                pos, atom_types, charge_types = property_classifier_guidance(
                    pos,
                    atom_types,
                    charge_types,
                    temb,
                    property_model,
                    batch,
                    num_atom_types=self.num_atom_types,
                    signal=signal,
                    guidance_scale=guidance_scale,
                    optimization="maximize",
                )

            elif importance_sampling and i % every_importance_t == 0 and importance_sampling_start <= i <= importance_sampling_end:
                node_feats_in = torch.cat([atom_types_joint, charge_types_joint], dim=-1)
                pos, node_feats_in, edge_index_global_lig, edge_attr_global_lig, \
                batch, _, num_nodes_lig = self.importance_sampling(node_feats_in=node_feats_in,
                                                                   pos=pos_joint,
                                                                   temb=temb,
                                                                   edge_index_local=None,
                                                                   edge_index_global=edge_index_global,
                                                                   edge_attr_global=edge_attr_global,
                                                                   batch=batch_full,
                                                                   batch_lig=batch,
                                                                   batch_edge_global=batch_edge_global,
                                                                   batch_num_nodes=num_nodes_lig,
                                                                   context=None,
                                                                   tau=tau,
                                                                   maximize_score=maximize_score,
                                                                   edge_index_global_lig=edge_index_global_lig,
                                                                   edge_attr_global_lig=edge_attr_global_lig,
                                                                   pocket_mask=pocket_mask,
                                                                   ca_mask=ca_mask,
                                                                   edge_mask=edge_mask,
                                                                   batch_pocket=batch_pocket,
                                                                   edge_mask_pocket=edge_mask_pocket,
                                                                   with_docking=with_docking,
                                                                   tau1=tau1,
                                                               )
                atom_types, charge_types = node_feats_in.split(
                    [self.num_atom_types, self.num_charge_classes], dim=-1
                )
                j, i = edge_index_global_lig
                mask = j < i
                mask_i = i[mask]
                
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

        # Move generated molecule back to the original pocket position for docking
        pos += pocket_cog[batch]
        pos_pocket += pocket_cog[batch_pocket]

        out_dict = {
            "coords_pred": pos,
            "coords_pocket": pos_pocket,
            "atoms_pred": atom_types,
            "atoms_pocket": atom_types_pocket,
            "charges_pred": charge_types,
            "bonds_pred": edge_attr_global_lig,
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
        if self.hparams.optimizer == "adam":
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.hparams["lr"],
                amsgrad=True,
                weight_decay=1.0e-12,
            )
        elif self.hparams.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                self.model.parameters(),
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
