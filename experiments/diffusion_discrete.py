import logging
import os
import sys
from datetime import datetime
from typing import List, Tuple

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from rdkit.Chem import RDConfig
from torch import Tensor
from torch_geometric.data import Batch
from torch_geometric.nn import radius_graph
from torch_geometric.utils import dense_to_sparse, sort_edge_index
from torch_scatter import scatter_mean
from tqdm import tqdm

sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))

from e3moldiffusion.coordsatomsbonds import (
    DenoisingEdgeNetwork,
    LatentEncoderNetwork,
    SoftMaxAttentionAggregation,
)
from e3moldiffusion.latent import PriorLatentLoss, get_latent_model
from e3moldiffusion.modules import (
    ClusterContinuousEmbedder,
    DenseLayer,
    GatedEquivBlock,
    TimestepEmbedder,
)
from e3moldiffusion.molfeat import get_bond_feature_dims
from experiments.data.abstract_dataset import AbstractDatasetInfos
from experiments.data.distributions import prepare_context
from experiments.data.utils import (
    get_fc_edge_index_with_offset,
    write_trajectory_as_xyz,
    write_xyz_file_from_batch,
)
from experiments.diffusion.categorical import CategoricalDiffusionKernel
from experiments.diffusion.continuous import DiscreteDDPM
from experiments.diffusion.utils import (
    bond_guidance,
    extract_func_groups_,
    extract_scaffolds_,
    initialize_edge_attrs_reverse,
    property_guidance_joint,
)
from experiments.losses import DiffusionLoss
from experiments.sampling.analyze import analyze_stability_for_molecules
from experiments.sampling.utils import calculate_sa
from experiments.utils import (
    coalesce_edges,
    get_lipinski_properties,
    get_molecules,
    load_bond_model,
    load_energy_model,
    load_latent_encoder,
    load_model,
    load_property_model,
    zero_mean,
)

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
        prop_dist=None,
        prop_norm=None,
        **kwargs,
    ):
        super().__init__()

        if "use_out_norm" not in hparams.keys():
            hparams["use_out_norm"] = True

        self.save_hyperparameters(hparams)
        self.i = 0
        self.mol_stab = 0.5

        self.dataset_info = dataset_info
        self.prop_norm = prop_norm
        self.prop_dist = prop_dist

        atom_types_distribution = dataset_info.atom_types.float()
        bond_types_distribution = dataset_info.edge_types.float()
        charge_types_distribution = dataset_info.charges_marginals.float()

        if (
            "use_absorbing_state" in self.hparams.keys()
        ):  # just as a hack for now for loading older models
            if self.hparams.use_absorbing_state:
                print(
                    "Using absorbing state instead of training distribution as prior!"
                )
                atom_types_distribution = torch.zeros((17,), dtype=torch.float32)
                atom_types_distribution[-1] = 1.0
                charge_types_distribution = torch.zeros((7,), dtype=torch.float32)
                charge_types_distribution[-1] = 1.0
                dataset_info.num_atom_types = 17
                dataset_info.num_charge_classes = 7

        self.register_buffer("atoms_prior", atom_types_distribution.clone())
        self.register_buffer("bonds_prior", bond_types_distribution.clone())
        self.register_buffer("charges_prior", charge_types_distribution.clone())

        self.hparams.num_atom_types = dataset_info.input_dims.X
        self.num_charge_classes = dataset_info.input_dims.C
        if "use_absorbing_state" in self.hparams.keys():  # just as a hack for now
            if self.hparams.use_absorbing_state:
                self.hparams.num_atom_types += 1
                self.num_charge_classes += 1

        self.remove_hs = hparams.get("remove_hs")
        if self.remove_hs:
            print("Model without modelling explicit hydrogens")

        self.num_atom_types = self.hparams.num_atom_types
        self.num_atom_features = self.num_atom_types + self.num_charge_classes
        self.num_bond_classes = 5

        self.smiles_list = smiles_list

        empirical_num_nodes = dataset_info.n_nodes
        self.register_buffer(name="empirical_num_nodes", tensor=empirical_num_nodes)

        if self.hparams.load_ckpt_from_pretrained is not None:
            print("Loading from pre-trained model checkpoint...")
            self.model = load_model(
                self.hparams.load_ckpt_from_pretrained, self.num_atom_features
            )
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
                coords_param=hparams["continuous_param"],
                use_pos_norm=hparams["use_pos_norm"],
                use_out_norm=hparams["use_out_norm"],
                ligand_pocket_interaction=hparams["ligand_pocket_interaction"],
                store_intermediate_coords=hparams["store_intermediate_coords"],
                distance_ligand_pocket=hparams["ligand_pocket_hidden_distance"],
                bond_prediction=hparams["bond_prediction"],
                property_prediction=hparams["property_prediction"],
                joint_property_prediction=hparams["joint_property_prediction"],
                regression_property=hparams["regression_property"],
                use_rbfs=hparams["use_rbfs"],
                mask_pocket_edges=hparams["mask_pocket_edges"],
                model_edge_rbf_interaction=hparams["model_edge_rbf_interaction"],
                model_global_edge=hparams["model_global_edge"],
            )

        self.max_nodes = dataset_info.max_n_nodes

        if self.hparams.use_centroid_context_embed:
            self.cluster_embed = ClusterContinuousEmbedder(
                self.hparams.num_context_features,
                self.hparams.latent_dim,
                dropout_prob=0.1,
            )
            self.t_embedder = TimestepEmbedder(self.hparams.latent_dim)

        if self.hparams.use_latent_encoder:
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
                    use_pos_norm=hparams["use_pos_norm_latent"],
                    use_out_norm=hparams["use_out_norm_latent"],
                )
                self.latent_lin = GatedEquivBlock(
                    in_dims=(hparams["sdim_latent"], hparams["vdim_latent"]),
                    out_dims=(hparams["latent_dim"], None),
                )
                self.graph_pooling = SoftMaxAttentionAggregation(
                    dim=hparams["latent_dim"]
                )
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
            self.bond_model.eval()

    def training_step(self, batch, batch_idx):
        return self.step_fnc(batch=batch, batch_idx=batch_idx, stage="train")

    def validation_step(self, batch, batch_idx):
        return self.step_fnc(batch=batch, batch_idx=batch_idx, stage="val")

    def on_validation_epoch_end(self):
        if self.hparams.dataset != "enamine":    
            if (self.current_epoch + 1) % self.hparams.test_interval == 0:
                if self.local_rank == 0:
                    print(f"Running evaluation in epoch {self.current_epoch + 1}")
                final_res = self.run_evaluation(
                    step=self.i,
                    dataset_info=self.dataset_info,
                    ngraphs=1000,
                    bs=self.hparams.inference_batch_size,
                    verbose=True,
                    inner_verbose=False,
                    eta_ddim=1.0,
                    ddpm=True,
                    every_k_step=1,
                    device="cuda" if self.hparams.gpus > 1 else "cpu",
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
                self.log(
                    name="val/mol_stable",
                    value=final_res.mol_stable[0],
                    on_epoch=True,
                    sync_dist=True,
                )
                self.log(
                    name="val/atm_stable",
                    value=final_res.atm_stable[0],
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
        polar_loss,
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
        
        if sa_loss != 0.0:
            self.log(
                f"{stage}/sa_loss",
                sa_loss,
                on_step=True,
                batch_size=batch_size,
                prog_bar=(stage == "train"),
                sync_dist=self.hparams.gpus > 1 and stage == "val",
            )

        if polar_loss != 0.0:
            self.log(
                f"{stage}/polar_loss",
                polar_loss,
                on_step=True,
                batch_size=batch_size,
                prog_bar=(stage == "train"),
                sync_dist=self.hparams.gpus > 1 and stage == "val",
            )
               
        if prior_loss != 0.0:
            self.log(
                f"{stage}/prior_loss",
                prior_loss,
                on_step=True,
                batch_size=batch_size,
                prog_bar=(stage == "train"),
                sync_dist=self.hparams.gpus > 1 and stage == "val",
            )
            
        if num_nodes_loss != 0.0:
            self.log(
                f"{stage}/num_nodes_loss",
                num_nodes_loss,
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

        if self.hparams.context_mapping:
            if self.hparams.use_lipinski_properties:
                context = get_lipinski_properties(batch.mol).to(self.device)
            else:
                context = prepare_context(
                    self.hparams["properties_list"],
                    self.prop_norm,
                    batch,
                    self.hparams.dataset,
                    inflate_batch=not self.hparams.use_centroid_context_embed,
                )
            batch.context = context

        out_dict = self(batch=batch, t=t)

        true_data = {
            "coords": (
                out_dict["coords_true"]
                if self.hparams.continuous_param == "data"
                else out_dict["coords_noise_true"]
            ),
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
        
        if self.hparams.joint_property_prediction:
            assert (
                "sa_score" in self.hparams.regression_property
                or "polarizability" in self.hparams.regression_property
            )
            if "sa_score" in self.hparams.regression_property:
                label_sa = (
                    torch.tensor([calculate_sa(mol) for mol in batch.mol])
                    .to(self.device)
                    .float()
                )
            else:
                label_sa = None
            if "polarizability" in self.hparams.regression_property:
                mean, mad = (
                    self.prop_dist.normalizer["polarizability"]["mean"],
                    self.prop_dist.normalizer["polarizability"]["mad"],
                )
                label_prop = (batch.y[:, -1].float() - mean) / mad
            else:
                label_prop = None
                
            true_data["properties"] = {"sa_score": label_sa, "property": label_prop}
            sa_pred, prop_pred = out_dict["property_pred"]
            pred_data["properties"] = {"sa_score": sa_pred, "property": prop_pred}
        else:
            true_data["properties"] = None
            pred_data["properties"] = None
            

        loss = self.diffusion_loss(
            true_data=true_data,
            pred_data=pred_data,
            batch=batch.batch,
            bond_aggregation_index=out_dict["bond_aggregation_index"],
            intermediate_coords=self.hparams.store_intermediate_coords
            and self.training,
            weights=weights,
        )

        final_loss = (
            self.hparams.lc_coords * loss["coords"]
            + self.hparams.lc_atoms * loss["atoms"]
            + self.hparams.lc_bonds * loss["bonds"]
            + self.hparams.lc_charges * loss["charges"]
            + self.hparams.lc_properties * loss["sa"]
            + self.hparams.lc_properties * loss["property"]
        )

        if self.hparams.use_latent_encoder:
            prior_loss = self.latentloss(inputdict=out_dict.get("latent"))
            num_nodes_loss = F.cross_entropy(
                out_dict["nodes"]["num_nodes_pred"], out_dict["nodes"]["num_nodes_true"]
            )
        else:
            prior_loss = 0.0
            num_nodes_loss = 0.0

        final_loss = (
            final_loss + self.hparams.prior_beta * prior_loss + num_nodes_loss
        )

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
            loss["sa"],
            loss["property"],
            prior_loss,
            num_nodes_loss,
            batch_size,
            stage,
        )

        return final_loss

    def encode_ligand(
        self,
        batch,
    ):
        atom_types = batch.x
        pos = batch.pos
        data_batch = batch.batch
        bond_edge_index = batch.edge_index
        bond_edge_attr = batch.edge_attr
        bond_edge_index, bond_edge_attr = sort_edge_index(
            edge_index=bond_edge_index, edge_attr=bond_edge_attr, sort_by_row=False
        )

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

    def forward(self, batch: Batch, t: Tensor):
        atom_types: Tensor = batch.x
        pos: Tensor = batch.pos
        charges: Tensor = batch.charges
        data_batch: Tensor = batch.batch
        bond_edge_index = batch.edge_index
        bond_edge_attr = batch.edge_attr
        context = batch.context if self.hparams.context_mapping else None
        n = batch.num_nodes
        bs = int(data_batch.max()) + 1
        bond_edge_index, bond_edge_attr = sort_edge_index(
            edge_index=bond_edge_index, edge_attr=bond_edge_attr, sort_by_row=False
        )

        # TIME EMBEDDING
        temb = t.float() / self.hparams.timesteps
        temb = temb.clamp(min=self.hparams.eps_min)
        temb = temb.unsqueeze(dim=1)

        z = None
        if self.hparams.use_centroid_context_embed:
            assert self.hparams.latent_dim is not None
            c1 = self.t_embedder(t)
            c2 = self.cluster_embed(context, self.training)
            if self.hparams.use_latent_encoder:
                c = c1 + c2
            else:
                z = c1 + c2
            context = None

        if self.hparams.use_latent_encoder:
            z = self.encode_ligand(batch)
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

            if self.hparams.use_centroid_context_embed:
                z = z + c
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

        pos_centered = zero_mean(pos, data_batch, dim=0, dim_size=bs)

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
        batch_edge_global = data_batch[edge_index_global[0]]

        # SAMPLING
        noise_coords_true, pos_perturbed = self.sde_pos.sample_pos(
            t, pos_centered, data_batch
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
        edge_attr_global_perturbed = (
            self.cat_bonds.sample_edges_categorical(
                t, edge_index_global, edge_attr_global, data_batch
            )
            if not self.hparams.bond_prediction
            else None
        )
        atom_feats_in_perturbed = torch.cat(
            [atom_types_perturbed, charges_perturbed], dim=-1
        )
        
        # edge-initial-features
        edge_initial_interaction = torch.zeros(
            (edge_index_global.size(1), 3),
            dtype=torch.float32,
            device=atom_types.device,
            )
        edge_initial_interaction[:, 0] = 1.0
         
        out = self.model(
            x=atom_feats_in_perturbed,
            t=temb,
            z=z,
            pos=pos_perturbed,
            edge_index_local=None,
            edge_index_global=edge_index_global,
            edge_attr_global=(
                edge_attr_global_perturbed if not self.hparams.bond_prediction else None
            ),
            batch=data_batch,
            batch_edge_global=batch_edge_global,
            context=context,
            edge_attr_initial_ohe=edge_initial_interaction,
        )

        out["coords_perturbed"] = pos_perturbed
        out["atoms_perturbed"] = atom_types_perturbed
        out["charges_perturbed"] = charges_perturbed
        out["bonds_perturbed"] = edge_attr_global_perturbed

        out["coords_true"] = pos_centered
        out["coords_noise_true"] = noise_coords_true
        out["atoms_true"] = atom_types.argmax(dim=-1)
        out["bonds_true"] = edge_attr_global
        out["charges_true"] = charges.argmax(dim=-1)

        if self.hparams.use_latent_encoder:
            out["latent"] = latentdict
            out["nodes"] = {
                "num_nodes_pred": pred_num_nodes,
                "num_nodes_true": true_num_nodes - 1,
            }

        out["bond_aggregation_index"] = edge_index_global[1]

        return out

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
        ddpm: bool = True,
        eta_ddim: float = 1.0,
        every_k_step: int = 1,
        run_test_eval: bool = False,
        save_traj: bool = False,
        fix_noise_and_nodes: bool = False,
        n_nodes: int = None,
        vary_n_nodes: bool = False,
        fixed_context: list = None,
        relax_sampling: bool = False,
        relax_steps: int = 10,
        ckpt_property_model: str = None,
        minimize_property: bool = False,
        classifier_guidance: bool = False,
        classifier_guidance_scale: float = 1.0e-4,
        classifier_guidance_start: int = 200,
        classifier_guidance_end: int = 300,
        every_guidance_t: int = 5,
        importance_sampling: bool = False,
        property_tau: float = 0.1,
        every_importance_t: int = 5,
        importance_sampling_start: int = 0,
        importance_sampling_end: int = 200,
        device: str = "cpu",
        renormalize_property: bool = True,
    ):

        b = ngraphs // bs
        l = [bs] * b
        if sum(l) != ngraphs:
            l.append(ngraphs - sum(l))
        assert sum(l) == ngraphs

        start = datetime.now()
        if verbose:
            if self.local_rank == 0:
                print(f"Creating {ngraphs} graphs in {l} batches")

        if self.hparams.use_latent_encoder:
            dataloader = (
                self.trainer.datamodule.val_dataloader()
                if not run_test_eval
                else self.trainer.datamodule.test_dataloader()
            )

        molecule_list = []
        for i, num_graphs in enumerate(l):
            if n_nodes is not None:
                fixed_num_nodes = (
                    torch.tensor(n_nodes).repeat(num_graphs).to(self.device)
                )
                if vary_n_nodes is not None:
                    fixed_num_nodes += torch.randint(
                        low=n_nodes - vary_n_nodes,
                        high=n_nodes + vary_n_nodes,
                        size=fixed_num_nodes.size(),
                    ).to(self.device)
            else:
                fixed_num_nodes = None
            molecules = self.reverse_sampling(
                num_graphs=num_graphs,
                verbose=inner_verbose,
                save_traj=save_traj,
                save_dir=save_dir,
                ddpm=ddpm,
                eta_ddim=eta_ddim,
                every_k_step=every_k_step,
                fix_noise_and_nodes=fix_noise_and_nodes,
                num_nodes=fixed_num_nodes,
                fixed_context=fixed_context,
                iteration=i,
                relax_sampling=relax_sampling,
                relax_steps=relax_steps,
                classifier_guidance=classifier_guidance,
                classifier_guidance_scale=classifier_guidance_scale,
                classifier_guidance_start=classifier_guidance_start,
                classifier_guidance_end=classifier_guidance_end,
                every_guidance_t=every_guidance_t,
                importance_sampling=importance_sampling,
                property_tau=property_tau,
                every_importance_t=every_importance_t,
                importance_sampling_start=importance_sampling_start,
                importance_sampling_end=importance_sampling_end,
                ckpt_property_model=ckpt_property_model,
                minimize_property=minimize_property,
                renormalize_property=renormalize_property,
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
            device=device,
        )

        save_cond = (
            self.mol_stab < stability_dict["mol_stable"]
            if self.hparams.dataset != "qm9"
            else (
                self.mol_stab < stability_dict["mol_stable"]
                and validity_dict["novelty"] > 0.75
            )
        )
        if save_cond and not run_test_eval:
            self.mol_stab = stability_dict["mol_stable"]
            save_path = os.path.join(self.hparams.save_dir, "best_mol_stab.ckpt")
            self.trainer.save_checkpoint(save_path)
            # for g in self.optimizers().param_groups:
            #     if g['lr'] > self.hparams.lr_min:
            #         g['lr'] *= 0.9

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
            return total_res, all_generated_smiles, stable_molecules
        else:
            return total_res

    @torch.no_grad()
    def generate_valid_samples(
        self,
        dataset_info,
        ngraphs: int = 4000,
        bs: int = 500,
        save_dir: str = None,
        return_molecules: bool = False,
        verbose: bool = False,
        inner_verbose=False,
        ddpm: bool = True,
        eta_ddim: float = 1.0,
        every_k_step: int = 1,
        use_energy_guidance: bool = False,
        ckpt_energy_model: str = None,
        save_traj: bool = False,
        fix_noise_and_nodes: bool = False,
        fixed_context: float = None,
        n_nodes: int = None,
        vary_n_nodes: int = None,
        relax_sampling: bool = False,
        relax_steps: int = 10,
        ckpt_property_model: str = None,
        classifier_guidance: bool = False,
        classifier_guidance_start: int = 200,
        classifier_guidance_end: int = 300,
        classifier_guidance_scale: float = 1.0e-4,
        every_guidance_t: int = 5,
        importance_sampling: bool = False,
        property_tau: float = 0.1,
        every_importance_t: int = 5,
        importance_sampling_start: int = 0,
        importance_sampling_end: int = 200,
        minimize_property: bool = True,
        device: str = "cpu",
        renormalize_property: bool = True,
    ):
        energy_model = None
        if use_energy_guidance:
            energy_model = load_energy_model(ckpt_energy_model, self.num_atom_features)
            # for param in self.energy_model.parameters():
            #    param.requires_grad = False
            energy_model.to(self.device)
            energy_model.eval()

        start = datetime.now()

        i = 0
        valid_molecule_list = []
        while len(valid_molecule_list) < ngraphs:
            b = ngraphs // bs
            l = [bs] * b
            if sum(l) != ngraphs:
                l.append(ngraphs - sum(l))
            assert sum(l) == ngraphs
            if verbose:
                if self.local_rank == 0:
                    print(f"Creating {ngraphs} graphs in {l} batches")
            molecule_list = []
            for num_graphs in l:
                if n_nodes is not None:
                    fixed_num_nodes = (
                        torch.tensor(n_nodes).repeat(num_graphs).to(self.device)
                    )
                    if vary_n_nodes is not None:
                        fixed_num_nodes += torch.randint(
                            low=n_nodes - vary_n_nodes,
                            high=n_nodes + vary_n_nodes,
                            size=fixed_num_nodes.size(),
                        ).to(self.device)
                else:
                    fixed_num_nodes = None
                molecules = self.reverse_sampling(
                    num_graphs=num_graphs,
                    verbose=inner_verbose,
                    save_traj=save_traj,
                    save_dir=save_dir,
                    ddpm=ddpm,
                    eta_ddim=eta_ddim,
                    every_k_step=every_k_step,
                    fix_noise_and_nodes=fix_noise_and_nodes,
                    num_nodes=fixed_num_nodes,
                    fixed_context=fixed_context,
                    iteration=i,
                    relax_sampling=relax_sampling,
                    relax_steps=relax_steps,
                    classifier_guidance=classifier_guidance,
                    classifier_guidance_start=classifier_guidance_start,
                    classifier_guidance_end=classifier_guidance_end,
                    classifier_guidance_scale=classifier_guidance_scale,
                    every_guidance_t=every_guidance_t,
                    importance_sampling=importance_sampling,
                    property_tau=property_tau,
                    every_importance_t=every_importance_t,
                    importance_sampling_start=importance_sampling_start,
                    importance_sampling_end=importance_sampling_end,
                    ckpt_property_model=ckpt_property_model,
                    minimize_property=minimize_property,
                    renormalize_property=renormalize_property,
                )

                molecule_list.extend(molecules)
                i += 1

            valid_molecules = analyze_stability_for_molecules(
                molecule_list=molecule_list,
                dataset_info=dataset_info,
                smiles_train=self.smiles_list,
                local_rank=self.local_rank,
                return_molecules=return_molecules,
                calculate_statistics=False,
                device=device,
            )
            valid_molecule_list.extend(valid_molecules)

        (
            stability_dict,
            validity_dict,
            statistics_dict,
            all_generated_smiles,
            stable_molecules,
            valid_molecules,
        ) = analyze_stability_for_molecules(
            molecule_list=valid_molecule_list,
            dataset_info=dataset_info,
            smiles_train=self.smiles_list,
            local_rank=self.local_rank,
            return_molecules=return_molecules,
            device=device,
        )

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

        total_res["run_time"] = str(run_time)
        total_res["ngraphs"] = str(len(valid_molecules))
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
            return total_res, all_generated_smiles, stable_molecules
        else:
            return total_res

    @torch.no_grad()
    def run_fixed_substructure_evaluation(
        self,
        dataset_info,
        save_dir: str = None,
        return_molecules: bool = False,
        verbose: bool = False,
        save_traj: bool = False,
        inner_verbose=False,
        ddpm: bool = True,
        eta_ddim: float = 1.0,
        every_k_step: int = 1,
        run_test_eval: bool = False,
        use_scaffold_dataset_sizes: bool = True,
        fraction_new_nodes: float = 0.0,
        scaffold_elaboration: bool = True,
        scaffold_hopping: bool = False,
        resample_steps: int = 0,
        max_num_batches: int = -1,
        relax_sampling: bool = False,
        relax_steps: int = 10,
        ckpt_property_model: str = None,
        classifier_guidance: bool = False,
        classifier_guidance_steps: int = 100,
        classifier_guidance_scale: float = 1.0e-4,
        importance_sampling: bool = False,
        property_tau: float = 0.1,
        every_importance_t: int = 5,
        importance_sampling_start: int = 0,
        importance_sampling_end: int = 200,
        minimize_property: bool = True,
        device: str = "cpu",
        renormalize_property: bool = True,
    ):

        g = torch.Generator()
        g.manual_seed(42)

        dataloader = (
            self.datamodule.val_dataloader()
            if not run_test_eval
            else self.datamodule.test_dataloader()
        )
        dataloader.generator = g

        molecule_list = []
        start = datetime.now()
        for i, batch_data in enumerate(dataloader):
            if (max_num_batches >= 0) and (i >= max_num_batches):
                break
            num_graphs = len(batch_data.batch.bincount())
            if use_scaffold_dataset_sizes:
                num_nodes = batch_data.batch.bincount().to(self.device)
            else:
                num_nodes = batch_data.batch.bincount().to(self.device)
                num_fixed = batch_data.fixed_nodes_mask.sum()
                num_nodes += torch.round(
                    (
                        torch.randint_like(num_nodes, 0, 100, dtype=num_nodes.dtype)
                        * num_fixed
                        * fraction_new_nodes
                    )
                    / 100
                )

            if scaffold_elaboration:
                extract_scaffolds_(batch_data)
            elif scaffold_hopping:
                extract_func_groups_(batch_data)
            else:
                raise Exception(
                    "Please specify which setting: Scaffold hopping or elaboration."
                )

            new_mols = self.reverse_sampling(
                num_graphs=num_graphs,
                verbose=inner_verbose,
                save_traj=save_traj,
                save_dir=save_dir,
                ddpm=ddpm,
                eta_ddim=eta_ddim,
                every_k_step=every_k_step,
                scaffold_elaboration=scaffold_elaboration,
                scaffold_hopping=scaffold_hopping,
                batch_data=batch_data,
                num_nodes=num_nodes,
                resample_steps=resample_steps,
                relax_sampling=relax_sampling,
                relax_steps=relax_steps,
                classifier_guidance=classifier_guidance,
                classifier_guidance_steps=classifier_guidance_steps,
                classifier_guidance_scale=classifier_guidance_scale,
                importance_sampling=importance_sampling,
                property_tau=property_tau,
                every_importance_t=every_importance_t,
                importance_sampling_start=importance_sampling_start,
                importance_sampling_end=importance_sampling_end,
                ckpt_property_model=ckpt_property_model,
                minimize_property=minimize_property,
                renormalize_property=renormalize_property,
            )

            for idx, mol in enumerate(new_mols):
                if scaffold_elaboration:
                    mol.fixed_mask = batch_data.fixed_nodes_mask[
                        batch_data.batch == idx
                    ].cpu()

                elif scaffold_hopping:
                    mol.fixed_mask = batch_data.fixed_nodes_mask[
                        batch_data.batch == idx
                    ].cpu()
                mol.ref_mol = batch_data.mol[idx]
                mol.trans_pos = batch_data.trans_pos[batch_data.batch == idx].cpu()
            molecule_list.extend(new_mols)

        import pickle

        with open(os.path.join(save_dir, "molecules.pkl"), "wb") as f:
            if verbose:
                if self.local_rank == 0:
                    print(f"saving mols to {os.path.join(save_dir, 'molecules.pkl')}")
            pickle.dump(molecule_list, f)

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
            calculate_statistics=True,
            device=device,
        )

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

        total_res["run_time"] = str(run_time)
        total_res["ngraphs"] = str(len(valid_molecules))
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
            return total_res, all_generated_smiles, stable_molecules
        else:
            return total_res

    def importance_sampling(
        self,
        node_feats_in,
        temb,
        pos,
        edge_index_local,
        edge_index_global,
        edge_attr_global,
        batch,
        batch_edge_global,
        context,
        batch_num_nodes,
        maximize_score: bool = True,
        sa_tau: float = 0.1,
        property_tau: float = 0.1,
        sa_model=None,
        property_model=None,
        kind: str = "polarizability",
        renormalize_property: bool = True,
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

        assert kind in ["polarizability"]

        if property_model is None:
            out = self.model(
                x=node_feats_in,
                t=temb,
                pos=pos,
                edge_index_local=edge_index_local,
                edge_index_global=edge_index_global,
                edge_attr_global=edge_attr_global,
                batch=batch,
                batch_edge_global=batch_edge_global,
                context=context,
            )
        elif kind == "polarizability" and property_model is not None:
            out = property_model(
                x=node_feats_in,
                t=temb,
                pos=pos,
                edge_index_local=edge_index_local,
                edge_index_global=edge_index_global,
                edge_attr_global=edge_attr_global,
                batch=batch,
                batch_edge_global=batch_edge_global,
                context=context,
            )

        prop_pred = out["property_pred"]
        _, prop = prop_pred

        if prop.dim() == 2:
            prop = prop.squeeze()

        if kind == "polarizability":
            if renormalize_property:
                prop = (
                    prop * self.prop_dist.normalizer["polarizability"]["mad"]
                    + self.prop_dist.normalizer["polarizability"]["mean"]
                )

        if not maximize_score:
            prop = -1.0 * prop

        n = pos.size(0)
        b = len(batch_num_nodes)

        weights_prop = (prop / property_tau).softmax(dim=0)

        select = torch.multinomial(
            weights_prop, num_samples=len(weights_prop), replacement=True
        )
        select = select.sort()[0]
        ptr = torch.concat(
            [
                torch.zeros((1,), device=batch_num_nodes.device, dtype=torch.long),
                batch_num_nodes.cumsum(0),
            ],
            dim=0,
        )

        batch_num_nodes_new = batch_num_nodes[select]
        # select
        batch_new = torch.arange(b, device=pos.device).repeat_interleave(
            batch_num_nodes_new
        )
        ## node level
        a, b = node_feats_in.size(1), pos.size(1)
        x = torch.concat([node_feats_in, pos], dim=1)
        x_split = x.split(batch_num_nodes.cpu().numpy().tolist(), dim=0)
        x_select = torch.concat([x_split[i] for i in select.cpu().numpy()], dim=0)
        node_feats_in, pos = x_select.split([a, b], dim=-1)

        ## edge level
        edge_slices = [
            slice(ptr[i - 1].item(), ptr[i].item()) for i in range(1, len(ptr))
        ]
        edge_slices_new = [edge_slices[i] for i in select.cpu().numpy()]

        # populate the dense edge-tensor
        E_dense = torch.zeros(
            (n, n, edge_attr_global.size(1)),
            dtype=edge_attr_global.dtype,
            device=edge_attr_global.device,
        )
        E_dense[edge_index_global[0], edge_index_global[1], :] = edge_attr_global

        # select the slices
        E_s = torch.stack(
            [
                torch.block_diag(*[E_dense[s, s, i] for s in edge_slices_new])
                for i in range(E_dense.size(-1))
            ],
            dim=-1,
        )
        new_ptr = torch.concat(
            [
                torch.zeros((1,), device=batch_num_nodes_new.device, dtype=torch.long),
                batch_num_nodes_new.cumsum(0),
            ],
            dim=0,
        )

        new_fc_edge_index = torch.concat(
            [
                get_fc_edge_index_with_offset(
                    n=batch_num_nodes_new[i].item(), offset=new_ptr[i].item()
                )
                for i in range(len(new_ptr) - 1)
            ],
            dim=1,
        )

        new_edge_attr = E_s[new_fc_edge_index[0], new_fc_edge_index[1], :]
        batch_edge_global = batch_new[new_fc_edge_index[0]]

        out = (
            pos.to(self.device),
            node_feats_in.to(self.device),
            new_fc_edge_index.to(self.device),
            new_edge_attr.to(self.device),
            batch_new.to(self.device),
            batch_edge_global.to(self.device),
            batch_num_nodes_new.to(self.device),
        )
        return out

    def reverse_sampling(
        self,
        num_graphs: int,
        verbose: bool = False,
        save_traj: bool = False,
        ddpm: bool = True,
        eta_ddim: float = 1.0,
        every_k_step: int = 1,
        energy_model=None,
        save_dir: str = None,
        fix_noise_and_nodes: bool = False,
        iteration: int = 0,
        chain_iterator=None,
        relax_sampling: bool = False,
        relax_steps: int = 10,
        scaffold_elaboration: bool = False,
        scaffold_hopping: bool = False,
        batch_data: Tensor = None,
        num_nodes: Tensor = None,
        fixed_context: float = None,
        resample_steps: int = 1,
        classifier_guidance: bool = False,
        classifier_guidance_start: int = 200,
        classifier_guidance_end: int = 300,
        every_guidance_t: int = 5,
        classifier_guidance_scale: float = 1.0e-4,
        importance_sampling: bool = False,
        property_tau: float = 0.1,
        every_importance_t: int = 5,
        importance_sampling_start: int = 0,
        importance_sampling_end: int = 200,
        ckpt_property_model: str = None,
        minimize_property: bool = True,
        renormalize_property: bool = True,
        data_batch=None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, List]:
        if num_nodes is not None:
            batch_num_nodes = num_nodes
        elif not fix_noise_and_nodes:
            batch_num_nodes = torch.multinomial(
                input=self.empirical_num_nodes,
                num_samples=num_graphs,
                replacement=True,
            ).to(self.device)
        else:
            seed_value = 42 + iteration
            torch.random.manual_seed(seed_value)
            batch_num_nodes = (
                torch.multinomial(
                    input=self.empirical_num_nodes,
                    num_samples=1,
                    replacement=True,
                )
                .repeat(num_graphs)
                .to(self.device)
            )
        batch_num_nodes = batch_num_nodes.clamp(min=1)
        batch = torch.arange(num_graphs, device=self.device).repeat_interleave(
            batch_num_nodes, dim=0
        )
        bs = int(batch.max()) + 1

        z = None
        context = None
        if self.prop_dist is not None:
            if fixed_context is not None:
                fixed_context = (
                    torch.tensor(fixed_context).unsqueeze(0).repeat(num_graphs, 1)
                )
                context = self.prop_dist.sample_fixed(fixed_context).to(self.device)
            else:
                context = self.prop_dist.sample_batch(batch_num_nodes).to(self.device)

            if self.hparams.use_centroid_context_embed:
                assert self.hparams.latent_dim is not None
                c2 = self.cluster_embed(context, self.training)
                if self.hparams.use_latent_encoder:
                    c = c2
                else:
                    z = c2
            context = context[batch]

        if self.hparams.use_latent_encoder:
            # encode ligand
            z = self.encode_ligand(data_batch.to(self.device))
            if self.hparams.use_centroid_context_embed:
                z = z + c

        if classifier_guidance and ckpt_property_model is not None:
            t = torch.arange(0, self.hparams.timesteps)
            alphas = self.sde_pos.alphas_cumprod[t]
            property_model = load_property_model(
                ckpt_property_model,
                self.num_atom_features,
                self.num_bond_classes,
                joint_prediction=True,  # HARDCODED, change if it is not a joint model
            )
            property_model.to(self.device)
            property_model.eval()

        if importance_sampling and ckpt_property_model is not None:
            property_model = load_property_model(
                ckpt_property_model,
                self.num_atom_features,
                self.num_bond_classes,
                joint_prediction=True,  # HARDCODED, change if it is not a joint model
            )
            property_model.to(self.device)
            property_model.eval()

        # initialiaze the 0-mean point cloud from N(0, I)
        pos = torch.randn(
            len(batch), 3, device=self.device, dtype=torch.get_default_dtype()
        )

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

        # edge_index_local = radius_graph(x=pos,
        #                                r=self.hparams.cutoff_local,
        #                                batch=batch,
        #                                max_num_neighbors=self.hparams.max_num_neighbors)
        edge_index_local = None
        edge_index_global = (
            torch.eq(batch.unsqueeze(0), batch.unsqueeze(-1)).int().fill_diagonal_(0)
        )
        edge_index_global, _ = dense_to_sparse(edge_index_global)
        edge_index_global = sort_edge_index(edge_index_global, sort_by_row=False)
        # edge-initial-features
        edge_initial_interaction = torch.zeros(
            (edge_index_global.size(1), 3),
            dtype=torch.float32,
            device=edge_index_global.device,
            )
        edge_initial_interaction[:, 0] = 1.0

        if scaffold_elaboration or scaffold_hopping:
            # global edge attributes
            edge_index_global, edge_attr_global = coalesce_edges(
                edge_index=edge_index_global,
                bond_edge_index=batch_data.edge_index.to(self.device),
                bond_edge_attr=batch_data.edge_attr.to(self.device),
                n=pos.size(0),
            )
            edge_index_global, orig_edge_attr_global = sort_edge_index(
                edge_index=edge_index_global,
                edge_attr=edge_attr_global,
                sort_by_row=False,
            )
            # global wbo
            edge_index_global, orig_wbo_global = coalesce_edges(
                edge_index=edge_index_global,
                bond_edge_index=batch_data.edge_index.to(self.device),
                bond_edge_attr=batch_data.wbo.to(self.device),
                n=pos.size(0),
            )
            edge_index_global, orig_wbo_global = sort_edge_index(
                edge_index=edge_index_global,
                edge_attr=orig_wbo_global,
                sort_by_row=False,
            )

        if not self.hparams.bond_prediction:
            (
                edge_attr_global,
                edge_index_global,
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
        batch_edge_global = batch[edge_index_global[0]]

        if scaffold_elaboration:
            fixed_nodes_mask = batch_data.scaffold_mask.to(self.device)
            batch_data.fixed_nodes_mask = fixed_nodes_mask.detach()
        elif scaffold_hopping:
            fixed_nodes_mask = batch_data.func_group_mask.to(self.device)
            batch_data.fixed_nodes_mask = fixed_nodes_mask.detach()

        if scaffold_elaboration or scaffold_hopping:
            # get original positions
            orig_pos = zero_mean(
                batch_data.pos, batch=batch_data.batch, dim_size=bs, dim=0
            ).to(self.device)
            batch_data.trans_pos = orig_pos.detach()
            # get original atom features
            orig_atom_types_int = batch_data.x.to(self.device)
            orig_atom_types_onehot = F.one_hot(
                orig_atom_types_int, self.num_atom_types
            ).float()
            orig_charge_types_int = batch_data.charges.to(self.device)
            orig_charge_types_onehot = self.dataset_info.one_hot_charges(
                orig_charge_types_int
            )
            orig_mulliken = batch_data.mulliken.to(self.device)
            # get fixed edge indices
            fixed_nodes_indices = torch.where(fixed_nodes_mask == True)[0]
            edge_0 = torch.where(
                edge_index_global[0][:, None] == fixed_nodes_indices[None, :]
            )[0]
            edge_1 = torch.where(
                edge_index_global[1][:, None] == fixed_nodes_indices[None, :]
            )[0]
            edge_index_between_fixed_nodes = edge_0[
                torch.where(edge_0[:, None] == edge_1[None, :])[0]
            ]
            edge_mask_between_fixed_nodes = torch.zeros_like(
                orig_edge_attr_global, dtype=torch.bool, device=self.device
            )
            edge_mask_between_fixed_nodes[edge_index_between_fixed_nodes] = True

        if chain_iterator is None:
            if self.hparams.continuous_param == "data":
                chain = range(0, self.hparams.timesteps)
            elif self.hparams.continuous_param == "noise":
                chain = range(0, self.hparams.timesteps - 1)

            chain = chain[::every_k_step]

            iterator = (
                tqdm(reversed(chain), total=len(chain)) if verbose else reversed(chain)
            )
        else:
            iterator = chain_iterator

        for i, timestep in enumerate(iterator):
            s = torch.full(
                size=(bs,), fill_value=timestep, dtype=torch.long, device=pos.device
            )
            t = s + 1

            if self.hparams.use_centroid_context_embed:
                z_t = z + self.t_embedder(t)

            temb = t / self.hparams.timesteps
            temb = temb.unsqueeze(dim=1)

            for r in range(resample_steps):
                node_feats_in = torch.cat([atom_types, charge_types], dim=-1)
                out = self.model(
                    x=node_feats_in,
                    z=z_t if self.hparams.use_centroid_context_embed else z,
                    t=temb,
                    pos=pos,
                    edge_index_local=edge_index_local,
                    edge_index_global=edge_index_global,
                    edge_attr_global=edge_attr_global,
                    batch=batch,
                    batch_edge_global=batch_edge_global,
                    context=context,
                    edge_initial_interaction=edge_initial_interaction,
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
                            s,
                            t,
                            pos,
                            coords_pred,
                            batch,
                            cog_proj=True,
                            eta_ddim=eta_ddim,
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
                else:
                    edge_attr_global = edges_pred

                if scaffold_elaboration or scaffold_hopping:
                    # create noised positions
                    _, pos_perturbed = self.sde_pos.sample_pos(
                        s, orig_pos, batch, remove_mean=True
                    )
                    # translate the COM of the masked noised nodes with the COM of the (masked) generated nodes
                    pos_perturbed_trans = (
                        pos_perturbed
                        + (
                            scatter_mean(
                                pos[fixed_nodes_mask], batch[fixed_nodes_mask], dim=0
                            )
                            - scatter_mean(
                                pos_perturbed[fixed_nodes_mask],
                                batch[fixed_nodes_mask],
                                dim=0,
                            )
                        )[batch]
                    )
                    # combine sampled pos of non fixed nodes with noised pos of fixed nodes
                    pos = (
                        pos * (~fixed_nodes_mask[:, None])
                        + pos_perturbed_trans * fixed_nodes_mask[:, None]
                    )

                    # create noised atom types
                    _, atom_types_perturbed = self.cat_atoms.sample_categorical(
                        s,
                        orig_atom_types_int,
                        batch,
                        self.dataset_info,
                        num_classes=self.num_atom_types,
                        type="atoms",
                    )
                    # combine sampled atom types of non fixed nodes with noised atom types of fixed nodes
                    atom_types = (
                        atom_types * (~fixed_nodes_mask[:, None])
                        + atom_types_perturbed * fixed_nodes_mask[:, None]
                    )

                    # create noised charges
                    _, charges_perturbed = self.cat_charges.sample_categorical(
                        s,
                        orig_charge_types_int,
                        batch,
                        self.dataset_info,
                        num_classes=self.num_charge_classes,
                        type="charges",
                    )
                    # combine sampled charges of non fixed nodes with noised charges of fixed nodes
                    charge_types = (
                        charge_types * (~fixed_nodes_mask[:, None])
                        + charges_perturbed * fixed_nodes_mask[:, None]
                    )

                    # create noised edges
                    edge_attr_global_perturbed = (
                        self.cat_bonds.sample_edges_categorical(
                            s,
                            edge_index_global,
                            orig_edge_attr_global,
                            batch,
                        )
                    )
                    # combine sampled edges of non fixed nodes with noised edges of fixed nodes
                    edge_attr_global = (
                        edge_attr_global * (~edge_mask_between_fixed_nodes[:, None])
                        + edge_attr_global_perturbed
                        * edge_mask_between_fixed_nodes[:, None]
                    )

                    if r < resample_steps and timestep > 0:
                        # noise the combined pos again
                        gamma_t, gamma_s = self.sde_pos.get_gamma(
                            t_int=t
                        ), self.sde_pos.get_gamma(t_int=t - 1)
                        (
                            sigma2_t_given_s,
                            sigma_t_given_s,
                            alpha_t_given_s,
                        ) = self.sde_pos.sigma_and_alpha_t_given_s(gamma_t, gamma_s)
                        bs = int(batch.max()) + 1
                        noise_coords_true = torch.randn_like(pos)
                        noise_coords_true = zero_mean(
                            noise_coords_true, batch=batch, dim_size=bs, dim=0
                        )
                        # get signal and noise coefficients for coords
                        pos = (
                            pos * alpha_t_given_s[batch][:, None]
                            + sigma_t_given_s[batch][:, None] * noise_coords_true
                        )

                        # noise the categorical atom features again
                        # Qtp1 = self.cat_atoms.Qt[t]
                        # probs = torch.einsum('nj, nji -> ni', [atom_types.argmax(dim=1), Qtp1]) # iwo batch
                        atom_types = self.cat_atoms.sample_categorical(
                            t,
                            atom_types.argmax(dim=1),
                            batch,
                            self.dataset_info,
                            num_classes=self.num_atom_types,
                            type="atoms",
                            cumulative=False,
                        )[1]
                        charge_types = self.cat_charges.sample_categorical(
                            t,
                            charge_types.argmax(dim=1)
                            - self.dataset_info.charge_offset,
                            batch,
                            self.dataset_info,
                            num_classes=self.num_charge_classes,
                            type="charges",
                            cumulative=False,
                        )[1]
                        # noise the edge attributes again
                        edge_attr_global = self.cat_bonds.sample_edges_categorical(
                            t,
                            edge_index_global,
                            edge_attr_global.argmax(dim=1),
                            batch,
                            cumulative=False,
                        )

                if self.hparams.bond_model_guidance:
                    pos = bond_guidance(
                        pos,
                        node_feats_in,
                        temb,
                        self.bond_model,
                        batch,
                        batch_edge_global,
                        edge_attr_global,
                        edge_index_local,
                        edge_index_global,
                    )

                if (
                    classifier_guidance
                    and property_model is not None
                    and i % every_guidance_t == 0
                    and classifier_guidance_start <= i <= classifier_guidance_end
                ):
                    signal = alphas[timestep] / (classifier_guidance_scale * 10)
                    (
                        pos,
                        node_feats_in,
                        edge_index_global,
                        edge_attr_global,
                        batch,
                        batch_edge_global,
                        batch_num_nodes,
                    ) = property_guidance_joint(
                        model=property_model,
                        node_feats_in=node_feats_in,
                        pos=pos,
                        temb=temb,
                        edge_index_local=edge_index_local,
                        edge_index_global=edge_index_global,
                        edge_attr_global=edge_attr_global,
                        batch=batch,
                        batch_edge_global=batch_edge_global,
                        batch_num_nodes=batch_num_nodes,
                        context=context,
                        minimize_property=minimize_property,
                        guidance_scale=classifier_guidance_scale,
                        signal=signal,
                    )

                # importance sampling
                if (
                    importance_sampling
                    and i % every_importance_t == 0
                    and importance_sampling_start <= i <= importance_sampling_end
                ):
                    node_feats_in = torch.cat([atom_types, charge_types], dim=-1)
                    (
                        pos,
                        node_feats_in,
                        edge_index_global,
                        edge_attr_global,
                        batch,
                        batch_edge_global,
                        batch_num_nodes,
                    ) = self.importance_sampling(
                        node_feats_in=node_feats_in,
                        pos=pos,
                        temb=temb,
                        edge_index_local=None,
                        edge_index_global=edge_index_global,
                        edge_attr_global=edge_attr_global,
                        batch=batch,
                        batch_edge_global=batch_edge_global,
                        batch_num_nodes=batch_num_nodes,
                        context=None,
                        maximize_score=not minimize_property,
                        property_tau=property_tau,
                        property_model=property_model,
                        kind="polarizability",
                        renormalize_property=renormalize_property,
                    )
                    atom_types, charge_types = node_feats_in.split(
                        [self.num_atom_types, self.num_charge_classes], dim=-1
                    )
                    j, i = edge_index_global
                    mask = j < i
                    mask_i = i[mask]

            if save_traj:
                atom_decoder = self.dataset_info.atom_decoder
                write_xyz_file_from_batch(
                    pos,
                    atom_types,
                    batch,
                    atom_decoder=atom_decoder,
                    path=os.path.join(save_dir, f"iter_{iteration}"),
                    i=i,
                )
                write_xyz_file_from_batch(
                    pos_perturbed_trans,
                    atom_types_perturbed,
                    batch,
                    atom_decoder=atom_decoder,
                    path=os.path.join(save_dir, f"noised_iter_{iteration}"),
                    i=i,
                )
        if relax_sampling:
            return self.relax_sampling(
                pos,
                atom_types,
                charge_types,
                edge_attr_global,
                edge_index_local,
                edge_index_global,
                batch,
                batch_edge_global,
                mask,
                mask_i,
                batch_size=bs,
                context=context,
                ddpm=ddpm,
                eta_ddim=eta_ddim,
                save_traj=save_traj,
                save_dir=save_dir,
                i=i,
                iteration=iteration,
                relax_steps=relax_steps,
            )

        out_dict = {
            "coords_pred": pos,
            "atoms_pred": atom_types,
            "charges_pred": charge_types,
            "bonds_pred": edge_attr_global,
        }
        molecules = get_molecules(
            out_dict,
            batch,
            edge_index_global,
            self.num_atom_types,
            self.num_charge_classes,
            self.dataset_info,
            device=self.device,
            mol_device="cpu",
            context=context,
            while_train=False,
        )

        if save_traj:
            write_trajectory_as_xyz(
                molecules,
                path=os.path.join(save_dir, f"iter_{iteration}"),
                strict=True,
            )

        return molecules

    def relax_sampling(
        self,
        pos,
        atom_types,
        charge_types,
        edge_attr_global,
        edge_index_local,
        edge_index_global,
        batch,
        batch_edge_global,
        mask,
        mask_i,
        batch_size,
        context,
        ddpm,
        eta_ddim,
        save_dir,
        save_traj,
        i,
        iteration,
        relax_steps=10,
    ):
        timestep = 0
        for j in range(relax_steps):
            s = torch.full(
                size=(batch_size,),
                fill_value=timestep,
                dtype=torch.long,
                device=pos.device,
            )
            t = s + 1

            temb = t / self.hparams.timesteps
            temb = temb.unsqueeze(dim=1)

            node_feats_in = torch.cat([atom_types, charge_types], dim=-1)
            out = self.model(
                x=node_feats_in,
                t=temb,
                pos=pos,
                edge_index_local=edge_index_local,
                edge_index_global=edge_index_global,
                edge_attr_global=edge_attr_global,
                batch=batch,
                batch_edge_global=batch_edge_global,
                context=context,
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
                        s,
                        t,
                        pos,
                        coords_pred,
                        batch,
                        cog_proj=True,
                        eta_ddim=eta_ddim,
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
                atom_decoder = self.dataset_info.atom_decoder
                write_xyz_file_from_batch(
                    pos,
                    atom_types,
                    batch,
                    atom_decoder=atom_decoder,
                    path=os.path.join(save_dir, f"iter_{iteration}"),
                    i=i + j,
                )

        out_dict = {
            "coords_pred": pos,
            "atoms_pred": atom_types,
            "charges_pred": charge_types,
            "bonds_pred": edge_attr_global,
        }

        molecules = get_molecules(
            out_dict,
            batch,
            edge_index_global,
            self.num_atom_types,
            self.num_charge_classes,
            self.dataset_info,
            device=self.device,
            mol_device="cpu",
            context=context,
            while_train=False,
        )

        if save_traj:
            write_trajectory_as_xyz(
                molecules,
                path=os.path.join(save_dir, f"iter_{iteration}"),
                strict=True,
            )

        return molecules

    def configure_optimizers(self):
        if self.hparams.use_latent_encoder:
            all_params = (
            list(self.model.parameters())
            + list(self.encoder.parameters())
            + list(self.latent_lin.parameters())
            + list(self.graph_pooling.parameters())
            + list(self.mu_logvar_z.parameters())
            + list(self.node_z.parameters())
        )
        else:
            all_params = list(self.model.parameters())
            
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
            "monitor": self.mol_stab,
            "strict": False,
        }
        return [optimizer], [scheduler]
