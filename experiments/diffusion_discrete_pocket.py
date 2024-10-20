import logging
import os
import pickle
import sys
from datetime import datetime
from typing import Dict, List, Tuple

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from rdkit.Chem import RDConfig
from torch import Tensor
from torch_geometric.data import Batch
from torch_geometric.nn import radius_graph
from torch_geometric.utils import dense_to_sparse, sort_edge_index
from torch_scatter import scatter_mean, scatter_add
from tqdm import tqdm

from scipy.optimize import linear_sum_assignment

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
    extract_scaffolds_from_batch,
    get_joint_edge_attrs,
    initialize_edge_attrs_reverse,
    property_guidance_lig_pocket,
)
from experiments.sampling.inpainting import get_edge_mask_inpainting
from experiments.losses import DiffusionLoss
from experiments.sampling.analyze import analyze_stability_for_molecules
from experiments.sampling.utils import calculate_sa
from experiments.utils import (
    CosineAnnealingWarmupRestarts,
    coalesce_edges,
    concat_ligand_pocket,
    get_lipinski_properties,
    get_molecules,
    load_bond_model,
    load_latent_encoder,
    load_model_ligand,
    load_property_model,
    remove_mean_pocket,
    pocket_clash_guidance
)
from experiments.xtb_energy import calculate_xtb_energy

sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))

from experiments.data.ligand.utils import get_space_size, sample_atom_num

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
            hparams["joint_property_prediction"] = 0
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
        if "use_out_norm" not in hparams.keys():
            hparams["use_out_norm"] = False
        if "dynamic_graph" not in hparams.keys():
            hparams["dynamic_graph"] = False

        if "knn" not in hparams.keys():
            hparams["knn"] = None
        if "hybrid_knn" not in hparams.keys():
            hparams["hybrid_knn"] = None
        if "knn_with_cutoff" not in hparams.keys():
            hparams["knn_with_cutoff"] = None
        if "use_rbfs" not in hparams.keys():
            hparams["use_rbfs"] = None
        if "dataset_cutoff" not in hparams.keys():
            hparams["dataset_cutoff"] = 5.0
        if "mask_pocket_edges" not in hparams.keys():
            hparams["mask_pocket_edges"] = False
        if "model_edge_rbf_interaction" not in hparams.keys():
            hparams["model_edge_rbf_interaction"] = False
        if "model_global_edge" not in hparams.keys():
            hparams["model_global_edge"] = False
        if "use_cutoff_damping" not in hparams.keys():
            hparams["use_cutoff_damping"] = False
        if "not_strict_ckpt" not in hparams.keys():
            hparams["not_strict_ckpt"] = False

        self.save_hyperparameters(hparams)

        self.knn = hparams["knn"]
        self.hybrid_knn = hparams["hybrid_knn"]
        self.knn_with_cutoff = hparams["knn_with_cutoff"]
        self.cutoff_p = hparams["cutoff_local"]
        self.cutoff_lp = hparams["cutoff_local"]

        self.i = 0
        self.validity = 0.0
        self.connected_components = 0.0
        self.angles_w1 = 1000.0
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
                strict=not self.hparams.not_strict_ckpt,
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
                dynamic_graph=hparams["dynamic_graph"],
                knn=hparams["knn"],
                hybrid_knn=hparams["hybrid_knn"],
                use_rbfs=hparams["use_rbfs"],
                mask_pocket_edges=hparams["mask_pocket_edges"],
                model_edge_rbf_interaction=hparams["model_edge_rbf_interaction"],
                model_global_edge=hparams["model_global_edge"],
                use_cutoff_damping=hparams["use_cutoff_damping"],
                use_centroid_context_embed=hparams["use_centroid_context_embed"],
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
            T=self.hparams.timesteps,
            enforce_zero_terminal_snr=False,
        )
        self.sde_bonds = DiscreteDDPM(
            beta_min=hparams["beta_min"],
            beta_max=hparams["beta_max"],
            N=hparams["timesteps"],
            scaled_reverse_posterior_sigma=True,
            schedule=self.hparams.noise_scheduler,
            nu=1.5,
            T=self.hparams.timesteps,
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
            prior_n_atoms=self.hparams.prior_n_atoms,
            n_nodes_bias=self.hparams.n_nodes_bias,
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
                except Exception:
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
                prior_n_atoms=(
                    "targetdiff" if not self.hparams.use_latent_encoder else "reference"
                ),
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
        property_loss,
        dloss,
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

        if sa_loss is not None:
            self.log(
                f"{stage}/sa_loss",
                sa_loss,
                on_step=True,
                batch_size=batch_size,
                prog_bar=(stage == "train"),
                sync_dist=self.hparams.gpus > 1 and stage == "val",
            )
        if property_loss != 0.0:
            self.log(
                f"{stage}/property_loss",
                property_loss,
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
        if prior_loss is not None:
            self.log(
                f"{stage}/prior_loss",
                prior_loss,
                on_step=True,
                batch_size=batch_size,
                prog_bar=(stage == "train"),
                sync_dist=self.hparams.gpus > 1 and stage == "val",
            )
        if num_nodes_loss is not None:
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

        molsize_weights = (
            batch.batch.bincount()
            if self.hparams.molsize_loss_weighting and stage == "train"
            else None
        )

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

        out_dict = self(batch=batch, t=t, latent_gamma=1.0)

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
                or "docking_score" in self.hparams.regression_property
                or "kiba_score" in self.hparams.regression_property
                or "ic50" in self.hparams.regression_property
            )
            if "sa_score" in self.hparams.regression_property:
                label_sa = (
                    torch.tensor([calculate_sa(mol) for mol in batch.mol])
                    .to(self.device)
                    .float()
                )
            else:
                label_sa = None
            if (
                "docking_score" in self.hparams.regression_property
                or "kiba_score" in self.hparams.regression_property
                or "ic50" in self.hparams.regression_property
            ):
                if "docking_score" in self.hparams.regression_property:
                    label_prop = batch.docking_scores.float()
                elif "kiba_score" in self.hparams.regression_property:
                    label_prop = batch.kiba_score.float()
                elif "ic50" in self.hparams.regression_property:
                    label_prop = batch.ic50.float()
                else:
                    raise Exception(
                        "Specified regression property not supported. Choose docking_score, kiba_score or ic50"
                    )
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
            batch=batch.pos_batch,
            bond_aggregation_index=out_dict["bond_aggregation_index"],
            intermediate_coords=self.hparams.store_intermediate_coords
            and self.training,
            weights=weights,
            molsize_weights=molsize_weights,
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
            final_loss = (
                final_loss + self.hparams.prior_beta * prior_loss + num_nodes_loss
            )
        else:
            prior_loss = num_nodes_loss = None

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
            dloss = self.dist_loss(dloss_true, dloss_pred)
            dloss = scatter_mean(dloss, ligand_i, dim=0)
            # apply loss weighting
            dloss = scatter_mean(dloss, batch.batch, dim=0)
            dloss = torch.sum(weights * dloss, dim=0)
            if self.hparams.ligand_pocket_hidden_distance:
                d_hidden = out_dict["dist_pred"]
                # latent loss
                dloss1 = self.dist_loss(dloss_true, d_hidden)
                # consistency loss between geometry and latent
                dloss2 = self.dist_loss(dloss_pred, d_hidden)
                # combine and apply loss weighting
                dloss12 = dloss1 + dloss2
                dloss12 = scatter_mean(dloss12, ligand_i, dim=0)
                dloss12 = scatter_mean(dloss12, batch.batch, dim=0)
                dloss12 = weights * dloss12
                dloss12 = torch.sum(dloss12, dim=0)
                dloss = dloss + dloss12
            final_loss = final_loss + 3.0 * dloss
        else:
            dloss = None

        if torch.any(final_loss.isnan()):
            final_loss = final_loss[~final_loss.isnan()]
            print(f"Detected NaNs. Terminating training at epoch {self.current_epoch}")
            exit()

        # if self.training:
        #     final_loss.backward()
        #     names_ = []
        #     for name, param in self.model.named_parameters():
        #         if param.grad is None:
        #             names_.append(name)
        #     import pdb

        #     pdb.set_trace()

        self._log(
            final_loss,
            loss["coords"],
            loss["atoms"],
            loss["charges"],
            loss["bonds"],
            loss["sa"],
            loss["property"],
            dloss,
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
        if self.hparams.use_scaffold_latent_embed:
            batch = extract_scaffolds_from_batch(batch).to(self.device)

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
            z = self.encode_ligand(batch.to(self.device))
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
            pred_num_nodes = self.node_z(z)
            true_num_nodes = batch.batch.bincount()
            latentdict = {
                "z_true": z,
                "z_pred": zpred,
                "mu": mu,
                "logvar": logvar,
                "w": w,
                "delta_log_pw": delta_log_pw,
            }

        pocket_noise = torch.randn_like(pos_pocket) * self.hparams.pocket_noise_std
        pos_pocket = pos_pocket + pocket_noise

        pos_centered, pos_centered_pocket = remove_mean_pocket(
            pos, pos_pocket, data_batch, data_batch_pocket
        )

        if self.hparams.flow_matching:
            (
                pos_perturbed,
                atom_types_perturbed,
                charges_perturbed,
                edge_attr_global_perturbed_lig,
            ) = self.sample_flow(
                pos_centered,
                atom_types,
                charges,
                bond_edge_index,
                bond_edge_attr,
                temb,
                data_batch,
            )

        else:
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
                atom_types = F.one_hot(
                    atom_types, num_classes=self.num_atom_types
                ).float()
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
                dense_edge[edge_index_global_lig[0, :], edge_index_global_lig[1, :]] = (
                    edge_attr_global_lig
                )
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
                assert (
                    torch.norm(noise_edges - noise_edges.permute(1, 0, 2)).item() == 0.0
                )

                signal = self.sde_bonds.sqrt_alphas_cumprod[t]
                std = self.sde_bonds.sqrt_1m_alphas_cumprod[t]

                signal_b = signal[data_batch].unsqueeze(-1).unsqueeze(-1)
                std_b = std[data_batch].unsqueeze(-1).unsqueeze(-1)
                dense_edge_ohe_perturbed = (
                    dense_edge_ohe * signal_b + noise_edges * std_b
                )

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
        # Concatenate Ligand-Pocket
        (
            pos_joint_perturbed,
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
        (
            edge_index_global,
            edge_attr_global_perturbed,
            batch_edge_global,
            edge_mask,
            edge_mask_pocket,
            edge_initial_interaction,
        ) = get_joint_edge_attrs(
            pos_perturbed,
            pos_centered_pocket,
            data_batch,
            data_batch_pocket,
            edge_attr_global_perturbed_lig,
            self.num_bond_classes,
            self.device,
            cutoff_p=self.cutoff_p,
            cutoff_lp=self.cutoff_lp,
            knn=self.knn,
            hybrid_knn=self.hybrid_knn,
            knn_with_cutoff=self.knn_with_cutoff,
            pocket_mask=pocket_mask,
        )

        # Concatenate all node features
        atom_feats_in_perturbed = torch.cat(
            [atom_types_perturbed, charges_perturbed], dim=-1
        )

        if self.hparams.use_latent_encoder:
            if self.training and self.hparams.dropout_prob > 0.0:
                # mask the latents
                p = torch.rand(z.size(0), device=z.device)
                p = (p > self.hparams.dropout_prob).unsqueeze(-1).float()
                z = z * p
                
        # if self.hparams.context_mapping:
        #     # mask context
        #     p = torch.rand(context.size(0), device=context.device)
        #     p = (p > self.hparams.dropout_prob).unsqueeze(-1).float()
        #     context = context * p
        #     context = context[data_batch]
        #     context_full = torch.zeros((batch_full.size(0), context.size(1)), device=context.device)
        #     context_full[:context.size(0)] = context
        #     del context
        #     context = context_full
        #     del context_full
        # else:
        #     context = None
         
        out = self.model(
            x=atom_feats_in_perturbed,
            z=z,
            t=temb,
            pos=pos_joint_perturbed,
            edge_index_local=None,
            edge_index_global=edge_index_global,
            edge_index_global_lig=edge_index_global_lig,
            edge_attr_global=(
                edge_attr_global_perturbed if not self.hparams.bond_prediction else None
            ),
            batch=batch_full,
            batch_edge_global=batch_edge_global,
            context=context,
            pocket_mask=pocket_mask.unsqueeze(1),
            edge_mask=edge_mask,
            edge_mask_pocket=edge_mask_pocket,
            batch_lig=data_batch,
            ca_mask=batch.pocket_ca_mask,
            batch_pocket=batch.pos_pocket_batch,
            edge_attr_initial_ohe=edge_initial_interaction,
            latent_gamma=latent_gamma,
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

        if self.hparams.use_latent_encoder:
            out["latent"] = latentdict
            out["nodes"] = {
                "num_nodes_pred": pred_num_nodes,
                "num_nodes_true": true_num_nodes - 1,
            }

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

    def sample_flow(
        self,
        pos_centered,
        atom_types,
        charges,
        bond_edge_index,
        bond_edge_attr,
        temb,
        data_batch,
    ):
        bs = len(data_batch.bincount())
        t1 = (
            torch.tensor(self.hparams.timesteps + 1, dtype=torch.long)
            .repeat(bs)
            .to(self.device)
        )
        t1 = t1.float() / self.hparams.timesteps

        _, x1_pos = self.sde_pos.sample_pos(
            t=t1,
            pos=pos_centered,
            data_batch=data_batch,
            remove_mean=False,
        )
        eps_pos = torch.randn_like(pos_centered)
        mu_t_pos = (1 - temb[data_batch]) * pos_centered + temb[data_batch] * x1_pos
        pos_perturbed = mu_t_pos + eps_pos * self.hparams.flow_matching_sigma

        # EDGES
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
        n = len(pos_centered)
        # create block diagonal matrix
        dense_edge = torch.zeros(n, n, device=self.device, dtype=torch.long)
        # populate entries with integer features
        dense_edge[edge_index_global_lig[0, :], edge_index_global_lig[1, :]] = (
            edge_attr_global_lig
        )
        dense_edge_ohe = (
            F.one_hot(dense_edge.view(-1, 1), num_classes=BOND_FEATURE_DIMS + 1)
            .view(n, n, -1)
            .float()
        )
        assert (
            torch.norm(dense_edge_ohe - dense_edge_ohe.permute(1, 0, 2)).item() == 0.0
        )

        # create symmetric noise for edge-attributes
        noise_edges = torch.randn_like(dense_edge_ohe)
        noise_edges = 0.5 * (noise_edges + noise_edges.permute(1, 0, 2))
        assert torch.norm(noise_edges - noise_edges.permute(1, 0, 2)).item() == 0.0

        signal = self.sde_bonds.sqrt_alphas_cumprod[t1]
        std = self.sde_bonds.sqrt_1m_alphas_cumprod[t1]

        signal_b = signal[data_batch].unsqueeze(-1).unsqueeze(-1)
        std_b = std[data_batch].unsqueeze(-1).unsqueeze(-1)
        dense_edge_ohe_perturbed = dense_edge_ohe * signal_b + noise_edges * std_b

        mu_t_edge = (1 - temb[data_batch]) * dense_edge + temb[
            data_batch
        ] * dense_edge_ohe_perturbed
        dense_edge_ohe_perturbed = (
            mu_t_edge + noise_edges * self.hparams.flow_matching_sigma
        )

        # retrieve as edge-attributes in PyG Format
        edge_attr_global_perturbed_lig = dense_edge_ohe_perturbed[
            edge_index_global_lig[0, :], edge_index_global_lig[1, :], :
        ]

        # ATOMS, CHARGES
        atom_types = F.one_hot(atom_types, num_classes=self.num_atom_types).float()
        charges = self.dataset_info.one_hot_charges(charges).float()
        _, x1_atoms = self.sde_atom_charge.sample(
            t=t1,
            feature=atom_types,
            data_batch=data_batch,
            remove_mean=False,
        )
        _, x1_charges = self.sde_atom_charge.sample(
            t=t1,
            feature=charges,
            data_batch=data_batch,
            remove_mean=False,
        )
        eps_atoms = torch.randn_like(atom_types)
        mu_t_atoms = (1 - temb[data_batch]) * atom_types + temb[data_batch] * x1_atoms
        atom_types_perturbed = mu_t_atoms + eps_atoms * self.hparams.flow_matching_sigma
        mu_t_charges = (1 - temb[data_batch]) * charges + temb[data_batch] * x1_charges
        charges_perturbed = (
            mu_t_charges + torch.randn_like(charges) * self.hparams.flow_matching_sigma
        )

        return (
            pos_perturbed,
            atom_types_perturbed,
            charges_perturbed,
            edge_attr_global_perturbed_lig,
        )

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
        property_classifier_guidance=None,
        property_classifier_guidance_complex=False,
        property_classifier_self_guidance=False,
        classifier_guidance_scale=None,
        ckpt_property_model: str = None,
        n_nodes_bias: int = 0,
        device: str = "cpu",
        encode_ligand: bool = True,
        prior_n_atoms: str = "targetdiff",
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
            if use_ligand_dataset_sizes or prior_n_atoms == "reference":
                num_nodes_lig = pocket_data.batch.bincount().to(self.device)
            elif prior_n_atoms == "conditional":
                num_nodes_lig = self.conditional_size_distribution.sample_conditional(
                    n1=None, n2=pocket_data.pos_pocket_batch.bincount()
                ).to(self.device)
                num_nodes_lig += n_nodes_bias
            elif prior_n_atoms == "targetdiff":
                _num_nodes_pockets = pocket_data.pos_pocket_batch.bincount()
                _pos_pocket_splits = pocket_data.pos_pocket.split(
                    _num_nodes_pockets.cpu().numpy().tolist(), dim=0
                )
                num_nodes_lig = torch.tensor(
                    [
                        sample_atom_num(
                            get_space_size(n.cpu().numpy()),
                            cutoff=self.hparams.dataset_cutoff,
                        )
                        for n in _pos_pocket_splits
                    ]
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
                ckpt_property_model=ckpt_property_model,
                property_classifier_guidance=property_classifier_guidance,
                property_classifier_guidance_complex=property_classifier_guidance_complex,
                property_classifier_self_guidance=property_classifier_self_guidance,
                classifier_guidance_scale=classifier_guidance_scale,
                save_dir=save_dir,
                build_obabel_mol=build_obabel_mol,
                iteration=i,
                encode_ligand=encode_ligand,
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
            save_cond = (
                save_cond or statistics_dict["sampling/AnglesW1"] < self.angles_w1
            )
        else:
            save_cond = False
        if save_cond:
            self.validity = validity_dict["validity"]
            self.connected_components = statistics_dict["connected_components"]
            self.angles_w1 = statistics_dict["sampling/AnglesW1"]
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
        build_obabel_mol=False,
        sanitize=True,
        every_k_step=1,
        fix_n_nodes=False,
        vary_n_nodes=False,
        n_nodes_bias=0,
        ckpt_property_model=None,
        ckpt_sa_model=None,
        ckpts_ensemble=None,
        property_classifier_guidance=None,
        property_classifier_guidance_complex=False,
        property_classifier_self_guidance=False,
        classifier_guidance_scale=None,
        classifier_guidance_kind: str = "sa_score",
        classifier_guidance_period: str = "all",
        sa_importance_sampling=False,
        sa_importance_sampling_start=0,
        sa_importance_sampling_end=200,
        sa_every_importance_t=5,
        sa_tau=0.1,
        property_importance_sampling: bool = False,
        property_importance_sampling_start=0,
        property_importance_sampling_end=200,
        property_every_importance_t=5,
        property_tau: float = 0.1,
        maximize_property=True,
        encode_ligand: bool = True,
        save_dir=None,
        prior_n_atoms: str = "conditional",
        joint_importance_sampling: bool = False,
        property_normalization: bool = False,
        latent_gamma: float = 1.0,
        use_lipinski_context: bool = True,
        context_fixed=None,
        clash_guidance: bool = False,
        clash_guidance_start=None,
        clash_guidance_end=None,
        clash_guidance_scale: float = 0.1,
        inpainting=False,
        emd_ot=False,
        importance_gradient_guidance=False,
    ):  
        
        if not inpainting:
            # DiffSBDD settings
            if prior_n_atoms == "conditional":

                if fix_n_nodes:
                    num_nodes_lig = pocket_data.batch.bincount().to(self.device)
                    if vary_n_nodes:
                        num_nodes_lig += torch.randint(
                            low=-n_nodes_bias // 2,
                            high=n_nodes_bias // 2,
                            size=num_nodes_lig.size(),
                        ).to(self.device)
                    else:
                        num_nodes_lig += n_nodes_bias

                else:

                    try:
                        pocket_size = pocket_data.pos_pocket_batch.bincount()[0].unsqueeze(
                            0
                        )
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
            # TargetDiff settings
            elif prior_n_atoms == "targetdiff":
                if "ligand_sizes" in pocket_data:
                    num_nodes_lig = pocket_data.ligand_sizes
                    if vary_n_nodes:
                        num_nodes_lig += torch.randint(
                            low=0, high=n_nodes_bias, size=num_nodes_lig.size()
                        ).to(self.device)
                    else:
                        num_nodes_lig += n_nodes_bias
                else:
                    if fix_n_nodes:
                        num_nodes_lig = pocket_data.batch.bincount().to(self.device)
                        if vary_n_nodes:
                            num_nodes_lig += torch.randint(
                                low=-n_nodes_bias // 2,
                                high=n_nodes_bias // 2,
                                size=num_nodes_lig.size(),
                            ).to(self.device)
                        else:
                            num_nodes_lig += n_nodes_bias

                    else:
                        _num_nodes_pockets = pocket_data.pos_pocket_batch.bincount()
                        _pos_pocket_splits = pocket_data.pos_pocket.split(
                            _num_nodes_pockets.cpu().numpy().tolist(), dim=0
                        )
                        num_nodes_lig = torch.tensor(
                            [
                                sample_atom_num(
                                    get_space_size(n.cpu().numpy()),
                                    cutoff=self.hparams.dataset_cutoff,
                                )
                                for n in _pos_pocket_splits
                            ]
                        ).to(self.device)

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
                relax_mol=relax_mol,
                max_relax_iter=max_relax_iter,
                sanitize=sanitize,
                build_obabel_mol=build_obabel_mol,
                ckpt_property_model=ckpt_property_model,
                ckpt_sa_model=ckpt_sa_model,
                ckpts_ensemble=ckpts_ensemble,
                property_classifier_guidance=property_classifier_guidance,
                property_classifier_guidance_complex=property_classifier_guidance_complex,
                property_classifier_self_guidance=property_classifier_self_guidance,
                classifier_guidance_scale=classifier_guidance_scale,
                classifier_guidance_kind=classifier_guidance_kind,
                classifier_guidance_period=classifier_guidance_period,
                sa_importance_sampling=sa_importance_sampling,
                sa_importance_sampling_start=sa_importance_sampling_start,
                sa_importance_sampling_end=sa_importance_sampling_end,
                sa_every_importance_t=sa_every_importance_t,
                sa_tau=sa_tau,
                property_importance_sampling=property_importance_sampling,
                property_importance_sampling_start=property_importance_sampling_start,
                property_importance_sampling_end=property_importance_sampling_end,
                property_every_importance_t=property_every_importance_t,
                property_tau=property_tau,
                maximize_property=maximize_property,
                save_dir=save_dir,
                encode_ligand=encode_ligand,
                joint_importance_sampling=joint_importance_sampling,
                property_normalization=property_normalization,
                latent_gamma=latent_gamma,
                use_lipinski_context=use_lipinski_context,
                context_fixed=context_fixed,
                clash_guidance=clash_guidance,
                clash_guidance_scale=clash_guidance_scale,
                clash_guidance_start=clash_guidance_start,
                clash_guidance_end=clash_guidance_end,
                importance_gradient_guidance=importance_gradient_guidance,
            )
        else:
            molecules = self.inpainting(
                pocket_data=pocket_data,
                num_graphs=num_graphs,
                verbose=inner_verbose,
                relax_mol=relax_mol,
                max_relax_iter=max_relax_iter,
                sanitize=sanitize,
                build_obabel_mol=build_obabel_mol,
                clash_guidance=clash_guidance,
                clash_guidance_scale=clash_guidance_scale,
                clash_guidance_start=clash_guidance_start,
                clash_guidance_end=clash_guidance_end,
                sa_importance_sampling=sa_importance_sampling,
                sa_importance_sampling_start=sa_importance_sampling_start,
                sa_importance_sampling_end=sa_importance_sampling_end,
                sa_every_importance_t=sa_every_importance_t,
                sa_tau=sa_tau,
                emd_ot=emd_ot,
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

    @torch.no_grad()
    def importance_sampling(
        self,
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
        maximize_score: bool = True,
        sa_tau: float = 0.1,
        property_tau: float = 0.1,
        kind: str = "sa_score",
        sa_model=None,
        property_model=None,
        ensemble_models=None,
        check_ensemble_variance=False,
        property_normalization=False,
        edge_attr_initial_ohe=None,
        z=None,
        latent_gamma=1.0,
        guidance_scale=0.1,
        importance_gradient_guidance=False,
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

        assert kind in ["sa_score", "docking_score", "ic50", "joint"]

        assert self.hparams.joint_property_prediction

        if len(ensemble_models) > 1:
            assert (
                len(ensemble_models) >= 2
            ), "Ensemble should consist of at least two models"
            preds = {"sa": [], "property": []}
            for ckpt in ensemble_models:
                property_model = load_property_model(
                    ckpt,
                    self.num_atom_features,
                    self.num_bond_classes,
                    joint_prediction=True,
                )
                property_model.to(pos.device)
                property_model.eval()
                out = property_model(
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
                    edge_attr_initial_ohe=edge_attr_initial_ohe,
                )
                sa, prop = out["property_pred"]
                if sa is not None:
                    preds["sa"].append(sa)
                if prop is not None:
                    preds["property"].append(prop)
            sa = (
                torch.cat(preds["sa"], dim=1).mean(dim=1).unsqueeze(1)
                if len(preds["sa"]) > 0
                else None
            )
            prop = (
                torch.cat(preds["property"], dim=1).mean(dim=1).unsqueeze(1)
                if len(preds["property"]) > 0
                else None
            )
            out["property_pred"] = (sa, prop)
            del property_model

        if len(ensemble_models) == 0 or (
            len(ensemble_models) > 1 and check_ensemble_variance
        ):
            model = (
                self.model
                if sa_model is None
                and property_model is None
                and (len(ensemble_models) == 0 or check_ensemble_variance)
                else (
                    sa_model
                    if kind == "sa_score" and sa_model is not None
                    else (
                        property_model
                        if (kind == "docking_score" or kind == "ic50")
                        and property_model is not None
                        else None
                    )
                )
            )
            
            if not importance_gradient_guidance:
                out = model(
                    x=node_feats_in,
                    t=temb,
                    z=z,
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
                    edge_attr_initial_ohe=edge_attr_initial_ohe,
                    latent_gamma=latent_gamma,
                )
            else:
                # gradient guidance
                pocket_mask = pocket_mask.bool()
                pos = pos.detach()
                pos_ligand = pos[pocket_mask].detach()
                pos_pocket = pos[~pocket_mask].detach()
                pos_pocket.requires_grad = False
                pos_ligand.requires_grad = True                
                with torch.enable_grad():
                    pos = torch.cat([pos_ligand, pos_pocket], dim=0)
                    out = model(
                    x=node_feats_in,
                    t=temb,
                    z=z,
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
                    edge_attr_initial_ohe=edge_attr_initial_ohe,
                    latent_gamma=latent_gamma,
                )
                
                if kind == "sa_score":
                    property_pred  = out["property_pred"][0]
                    sign = 1.0
                elif kind == "docking_score":
                    property_pred  = out["property_pred"][1]
                    sign = -1.0
                elif kind == "ic50":
                    property_pred  = out["property_pred"][1]
                    sign = 1.0
                        
                grad_outputs = [torch.ones_like(property_pred)]
                grad_shift = torch.autograd.grad(
                [property_pred],
                [pos_ligand],
                grad_outputs=grad_outputs,
                create_graph=False,
                retain_graph=False,
                )[0]
                
                pos_ligand = pos_ligand + sign * guidance_scale * grad_shift[:, :3]
                pos_ligand.detach_()
                pos = torch.cat([pos_ligand, pos_pocket], dim=0)
                
        pocket_mask = pocket_mask.bool()
        node_feats_in = node_feats_in[pocket_mask]
        pos = pos[pocket_mask]
        prop_pred = out["property_pred"]
        sa, prop = prop_pred
        sa, prop = sa.detach(), prop.detach()
        sa = (
            sa.squeeze(1).sigmoid()
            if sa is not None and (kind == "sa_score" or kind == "joint")
            else None
        )
        if prop is not None and (
            kind == "docking_score" or kind == "ic50" or kind == "joint"
        ):
            if prop.dim() == 2:
                prop = prop.squeeze()
            if kind == "docking_score":
                prop = -1.0 * prop
                if property_normalization:
                    N = batch_lig.bincount().float()
                    prop = prop / torch.sqrt(N)
        else:
            prop = None

        if not maximize_score and sa is not None:
            sa = 1.0 - sa

        n = pos.size(0)
        b = len(batch_num_nodes)

        weights_sa = (sa / sa_tau).softmax(dim=0) if sa is not None else None

        weights_prop = (
            (prop / property_tau).softmax(dim=0) if prop is not None else None
        )

        if kind == "joint":
            assert sa is not None and prop is not None
            # weights_add = 1.0 * (weights_sa + weights_prop)
            weights_add = weights_prop  # 0.0
            weights_mul = weights_sa * weights_prop
            weights = 1.0 * (weights_add + weights_mul)
            weights = weights.softmax(dim=0)
        elif kind == "sa_score":
            assert sa is not None
            weights = weights_sa
        elif kind == "docking_score" or kind == "ic50":
            assert prop is not None
            weights = weights_prop

        select = torch.multinomial(weights, num_samples=len(weights), replacement=True)
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
            (n, n, edge_attr_global_lig.size(1)),
            dtype=edge_attr_global_lig.dtype,
            device=edge_attr_global_lig.device,
        )
        E_dense[edge_index_global_lig[0], edge_index_global_lig[1], :] = (
            edge_attr_global_lig
        )

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
        # batch_edge_global = batch_new[new_fc_edge_index[0]]
        # batch_edge_global = None

        out = (
            pos.to(self.device),
            node_feats_in.to(self.device),
            new_fc_edge_index.to(self.device),
            new_edge_attr.to(self.device),
            batch_new.to(self.device),
            {"batch_num_nodes_old": batch_num_nodes, "select": select},
            batch_num_nodes_new.to(self.device),
        )
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
        save_dir: str = None,
        relax_mol=False,
        max_relax_iter=200,
        sanitize=False,
        build_obabel_mol=False,
        iteration: int = 0,
        ckpt_property_model=None,
        ckpt_sa_model=None,
        ckpts_ensemble=None,
        property_classifier_guidance=None,
        property_classifier_guidance_complex=False,
        property_classifier_self_guidance=False,
        classifier_guidance_scale=None,
        classifier_guidance_kind: str = "sa_score",
        classifier_guidance_period: str = "all",
        sa_importance_sampling=False,
        sa_importance_sampling_start=0,
        sa_importance_sampling_end=200,
        sa_every_importance_t=5,
        sa_tau=0.1,
        property_importance_sampling: bool = False,
        property_importance_sampling_start=0,
        property_importance_sampling_end=200,
        property_every_importance_t=5,
        property_tau: float = 0.1,
        maximize_property=True,
        encode_ligand: bool = True,
        joint_importance_sampling=False,
        property_normalization=False,
        latent_gamma: float = 1.0,
        use_lipinski_context=True, ## placeholder only used in inference as of now
        context_fixed=None,
        clash_guidance: bool = False,
        clash_guidance_start=None,
        clash_guidance_end=None,
        clash_guidance_scale: float = 0.1,
        importance_gradient_guidance=False,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, List]:
        
        pos_pocket = pocket_data.pos_pocket.to(self.device)
        batch_pocket = pocket_data.pos_pocket_batch.to(self.device)
        x_pocket = pocket_data.x_pocket.to(self.device)

        try:
            ca_mask = pocket_data.ca_mask.to(self.device)
        except Exception:
            ca_mask = None

        batch = torch.arange(num_graphs, device=self.device).repeat_interleave(
            num_nodes_lig, dim=0
        )
        bs = int(batch.max()) + 1

        z = None
        # Assumption that context enters on the graph-level, e.g., for the ligand a global variable like logp, sa, or qed.
        if self.hparams.context_mapping:
            # sample context condition
            z = None
            context = None
            if self.prop_dist is not None and not self.hparams.use_centroid_context_embed:
                context = self.prop_dist.sample_batch(num_nodes_lig).to(self.device)[batch]
            elif self.hparams.use_centroid_context_embed:
                assert self.hparams.latent_dim is not None
                if self.hparams.use_lipinski_properties:
                    context = get_lipinski_properties(pocket_data.mol).to(self.device)
                    if context_fixed is not None:
                        if context_fixed.ndim==1:
                            context_fixed = context_fixed.unsqueeze(0)
                        assert context_fixed.size(1) == context.size(1)
                        context[:, ] = context_fixed
                c = self.cluster_embed(context, train=False)
                if not self.hparams.use_latent_encoder:
                    z = c
                context = None
            else:
                if self.hparams.use_lipinski_properties:
                    context = get_lipinski_properties(pocket_data.mol).to(self.device)
                    if context_fixed is not None:
                        if context_fixed.ndim==1:
                            context_fixed = context_fixed.unsqueeze(0)
                        assert context_fixed.size(1) == context.size(1)
                        context[:, ] = context_fixed
        else:
            context = None
        
        if self.hparams.use_latent_encoder:
            if self.hparams.latentmodel == "mmd":
                assert encode_ligand
            if encode_ligand:
                # encode ligand
                z = self.encode_ligand(pocket_data.to(self.device))
                if self.hparams.use_centroid_context_embed:
                    z = z + c
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
                if self.hparams.use_centroid_context_embed:
                    z = z + c

        if (
            property_classifier_self_guidance
            or property_classifier_guidance
            or property_classifier_guidance_complex
        ):
            t = torch.arange(0, self.hparams.timesteps)
            alphas = self.sde_pos.alphas_cumprod[t]
        if (
            property_classifier_guidance
            or property_classifier_guidance_complex
            or property_importance_sampling
        ) and ckpt_property_model is not None:
            property_model = load_property_model(
                ckpt_property_model,
                self.num_atom_features,
                self.num_bond_classes,
                joint_prediction=property_classifier_guidance_complex
                or property_importance_sampling,
            )
            property_model.to(self.device)
            property_model.eval()
        else:
            property_model = None
        if sa_importance_sampling and ckpt_sa_model is not None:
            sa_model = load_property_model(
                ckpt_sa_model,
                self.num_atom_features,
                self.num_bond_classes,
                joint_prediction=sa_importance_sampling,
            )
            sa_model.to(self.device)
            sa_model.eval()
        else:
            sa_model = None

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
        (
            edge_index_global,
            edge_attr_global,
            batch_edge_global,
            edge_mask,
            edge_mask_pocket,
            edge_initial_interaction,
        ) = get_joint_edge_attrs(
            pos,
            pos_pocket,
            batch,
            batch_pocket,
            edge_attr_global_lig,
            self.num_bond_classes,
            self.device,
            cutoff_p=self.cutoff_p,
            cutoff_lp=self.cutoff_lp,
            knn=self.knn,
            hybrid_knn=self.hybrid_knn,
            knn_with_cutoff=self.knn_with_cutoff,
            pocket_mask=pocket_mask,
        )

        if self.hparams.continuous_param == "data":
            chain = range(0, self.hparams.timesteps)
        elif self.hparams.continuous_param == "noise":
            chain = range(0, self.hparams.timesteps - 1)

        chain = chain[::every_k_step]

        iterator = (
            tqdm(reversed(chain), total=len(chain)) if verbose else reversed(chain)
        )
        
        if context is not None and not self.hparams.use_latent_encoder: # classifier-free guidance
            context_full = torch.zeros((pos_joint.size(0), context.size(1)), device=self.device)
            # node-slicing creating copies of global context for each node in the graph(s)
            context = context[batch]
            context_full[:context.size(0)] = context
            del context
            context = context_full
            del context_full
            
            if not use_lipinski_context:
                context = torch.zeros_like(context)
        
        if clash_guidance_start is None:
            clash_guidance_start = 0
        if clash_guidance_end is None:
            clash_guidance_end = self.hparams.timesteps
            
        for i, timestep in enumerate(iterator):
            s = torch.full(
                size=(bs,), fill_value=timestep, dtype=torch.long, device=self.device
            )
            t = s + 1

            temb = t / self.hparams.timesteps
            temb = temb.unsqueeze(dim=1)

            node_feats_in = torch.cat([atom_types_joint, charge_types_joint], dim=-1)

            out = self.model(
                x=node_feats_in,
                z=(
                    z + self.t_embedder(t)
                    if self.hparams.use_centroid_context_embed
                    else z
                ),
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
                edge_attr_initial_ohe=edge_initial_interaction,
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
            
            if clash_guidance:
                if clash_guidance_start <= i <= clash_guidance_end:
                    _, delta = pocket_clash_guidance(x_l=pos, x_p=pos_pocket,
                                                    batch_l=batch, batch_p=batch_pocket,
                                                    sigma=2.0
                                                    )
                    pos = pos + clash_guidance_scale * delta
                
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
                edge_initial_interaction,
            ) = get_joint_edge_attrs(
                pos,
                pos_pocket,
                batch,
                batch_pocket,
                edge_attr_global_lig,
                self.num_bond_classes,
                self.device,
                cutoff_p=self.cutoff_p,
                cutoff_lp=self.cutoff_lp,
                knn=self.knn,
                hybrid_knn=self.hybrid_knn,
                knn_with_cutoff=self.knn_with_cutoff,
                pocket_mask=pocket_mask,
            )

            if (
                property_classifier_self_guidance
                or property_classifier_guidance_complex
            ):  
                                
                if "sa_score" in classifier_guidance_kind:
                    check = ((classifier_guidance_period == "all") or (i % sa_every_importance_t == 0 
                                                                       and sa_importance_sampling_start
                                                                       <= i
                                                                       <= sa_importance_sampling_end)
                             )
                    if check:
                        (
                            pos,
                            atom_types,
                            charge_types,
                        ) = property_guidance_lig_pocket(
                            model=(
                                self.model
                                if property_classifier_self_guidance
                                else property_model
                            ),
                            pos=pos,
                            pos_pocket=pos_pocket,
                            atom_types=atom_types,
                            atom_types_pocket=atom_types_pocket,
                            charge_types=charge_types,
                            charges_pocket=charges_pocket,
                            edge_index_global=edge_index_global,
                            edge_index_global_lig=edge_index_global_lig,
                            edge_attr_global=edge_attr_global,
                            edge_initial_interaction=edge_initial_interaction,
                            batch=batch,
                            batch_pocket=batch_pocket,
                            batch_full=batch_full,
                            batch_edge_global=batch_edge_global,
                            pocket_mask=pocket_mask,
                            edge_mask=edge_mask,
                            edge_mask_pocket=edge_mask_pocket,
                            ca_mask=pocket_data.pocket_ca_mask.to(batch.device),
                            num_atom_types=self.num_atom_types,
                            atoms_continuous=self.hparams.atoms_continuous,
                            temb=temb,
                            context=context,
                            guidance_scale=classifier_guidance_scale,
                            minimize_property=False,
                        )
                        
                if "docking_score" in classifier_guidance_kind:
                    check = ((classifier_guidance_period == "all") or (i % property_every_importance_t == 0 
                                                                       and property_importance_sampling_start
                                                                       <= i
                                                                       <= property_importance_sampling_end)
                             )
                    if check:
                        (
                            pos,
                            atom_types,
                            charge_types,
                        ) = property_guidance_lig_pocket(
                            model=(
                                self.model
                                if property_classifier_self_guidance
                                else property_model
                            ),
                            pos=pos,
                            pos_pocket=pos_pocket,
                            atom_types=atom_types,
                            atom_types_pocket=atom_types_pocket,
                            charge_types=charge_types,
                            charges_pocket=charges_pocket,
                            edge_index_global=edge_index_global,
                            edge_index_global_lig=edge_index_global_lig,
                            edge_attr_global=edge_attr_global,
                            edge_initial_interaction=edge_initial_interaction,
                            batch=batch,
                            batch_pocket=batch_pocket,
                            batch_full=batch_full,
                            batch_edge_global=batch_edge_global,
                            pocket_mask=pocket_mask,
                            edge_mask=edge_mask,
                            edge_mask_pocket=edge_mask_pocket,
                            ca_mask=pocket_data.pocket_ca_mask.to(batch.device),
                            num_atom_types=self.num_atom_types,
                            atoms_continuous=self.hparams.atoms_continuous,
                            temb=temb,
                            context=context,
                            guidance_scale=classifier_guidance_scale,
                            minimize_property=True,
                        )
                        
            elif property_classifier_guidance:
                signal = 1.0  # alphas[timestep] / (guidance_scale * 10)
                pos, atom_types, charge_types = property_classifier_guidance(
                    pos,
                    atom_types,
                    charge_types,
                    temb,
                    property_model,
                    batch,
                    num_atom_types=self.num_atom_types,
                    signal=signal,
                    guidance_scale=classifier_guidance_scale,
                    optimization="minimize",
                )

            elif (
                sa_importance_sampling
                and i % sa_every_importance_t == 0
                and sa_importance_sampling_start <= i <= sa_importance_sampling_end
            ):
                node_feats_in = torch.cat(
                    [atom_types_joint, charge_types_joint], dim=-1
                )
                (
                    pos,
                    node_feats_in,
                    edge_index_global_lig,
                    edge_attr_global_lig,
                    batch,
                    _,
                    num_nodes_lig,
                ) = self.importance_sampling(
                    node_feats_in=node_feats_in,
                    pos=pos_joint,
                    temb=temb,
                    z=z,
                    edge_index_local=None,
                    edge_index_global=edge_index_global,
                    edge_attr_global=edge_attr_global,
                    batch=batch_full,
                    batch_lig=batch,
                    batch_edge_global=batch_edge_global,
                    batch_num_nodes=num_nodes_lig,
                    context=None,
                    sa_tau=sa_tau,
                    maximize_score=True,
                    edge_index_global_lig=edge_index_global_lig,
                    edge_attr_global_lig=edge_attr_global_lig,
                    pocket_mask=pocket_mask,
                    ca_mask=ca_mask,
                    edge_mask=edge_mask,
                    batch_pocket=batch_pocket,
                    edge_mask_pocket=edge_mask_pocket,
                    kind="sa_score",
                    sa_model=sa_model,
                    ensemble_models=ckpts_ensemble,
                    property_normalization=False,
                    edge_attr_initial_ohe=edge_initial_interaction,
                    latent_gamma=latent_gamma,
                    importance_gradient_guidance=importance_gradient_guidance,
                )
                atom_types, charge_types = node_feats_in.split(
                    [self.num_atom_types, self.num_charge_classes], dim=-1
                )
                jj, ii = edge_index_global_lig
                mask = jj < ii
                mask_i = ii[mask]

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
            (
                edge_index_global,
                edge_attr_global,
                batch_edge_global,
                edge_mask,
                edge_mask_pocket,
                edge_initial_interaction,
            ) = get_joint_edge_attrs(
                pos,
                pos_pocket,
                batch,
                batch_pocket,
                edge_attr_global_lig,
                self.num_bond_classes,
                self.device,
                cutoff_p=self.cutoff_p,
                cutoff_lp=self.cutoff_lp,
                knn=self.knn,
                hybrid_knn=self.hybrid_knn,
                knn_with_cutoff=self.knn_with_cutoff,
                pocket_mask=pocket_mask,
            )

            if (
                property_importance_sampling
                and i % property_every_importance_t == 0
                and property_importance_sampling_start
                <= i
                <= property_importance_sampling_end
                and not joint_importance_sampling
            ):
                node_feats_in = torch.cat(
                    [atom_types_joint, charge_types_joint], dim=-1
                )
                (
                    pos,
                    node_feats_in,
                    edge_index_global_lig,
                    edge_attr_global_lig,
                    batch,
                    _,
                    num_nodes_lig,
                ) = self.importance_sampling(
                    node_feats_in=node_feats_in,
                    pos=pos_joint,
                    temb=temb,
                    z=z,
                    edge_index_local=None,
                    edge_index_global=edge_index_global,
                    edge_attr_global=edge_attr_global,
                    batch=batch_full,
                    batch_lig=batch,
                    batch_edge_global=batch_edge_global,
                    batch_num_nodes=num_nodes_lig,
                    context=None,
                    maximize_score=maximize_property,
                    edge_index_global_lig=edge_index_global_lig,
                    edge_attr_global_lig=edge_attr_global_lig,
                    pocket_mask=pocket_mask,
                    ca_mask=ca_mask,
                    edge_mask=edge_mask,
                    batch_pocket=batch_pocket,
                    edge_mask_pocket=edge_mask_pocket,
                    property_tau=property_tau,
                    kind=(
                        property_model.regression_property[-1]
                        if property_model is not None
                        else self.hparams.regression_property[-1]
                    ),  # currently hardcoded for max. two properties, whereby SA is always the first and the property the last argument!
                    property_model=property_model,
                    ensemble_models=ckpts_ensemble,
                    property_normalization=property_normalization,
                    edge_attr_initial_ohe=edge_initial_interaction,
                    latent_gamma=latent_gamma,
                    importance_gradient_guidance=importance_gradient_guidance,
                )
                atom_types, charge_types = node_feats_in.split(
                    [self.num_atom_types, self.num_charge_classes], dim=-1
                )
                jj, ii = edge_index_global_lig
                mask = jj < ii
                mask_i = ii[mask]

            if (
                joint_importance_sampling
                and i % property_every_importance_t == 0
                and property_importance_sampling_start
                <= i
                <= property_importance_sampling_end
                and i % sa_every_importance_t == 0
                and sa_importance_sampling_start <= i <= sa_importance_sampling_end
            ):

                # print("Joint importance sampling")
                # SA should act as filter overlaying the property importance sampling
                assert sa_importance_sampling and property_importance_sampling

                node_feats_in = torch.cat(
                    [atom_types_joint, charge_types_joint], dim=-1
                )

                (
                    pos,
                    node_feats_in,
                    edge_index_global_lig,
                    edge_attr_global_lig,
                    batch,
                    _,
                    num_nodes_lig,
                ) = self.importance_sampling(
                    node_feats_in=node_feats_in,
                    pos=pos_joint,
                    temb=temb,
                    z=z,
                    edge_index_local=None,
                    edge_index_global=edge_index_global,
                    edge_attr_global=edge_attr_global,
                    batch=batch_full,
                    batch_lig=batch,
                    batch_edge_global=batch_edge_global,
                    batch_num_nodes=num_nodes_lig,
                    context=None,
                    maximize_score=maximize_property,
                    edge_index_global_lig=edge_index_global_lig,
                    edge_attr_global_lig=edge_attr_global_lig,
                    pocket_mask=pocket_mask,
                    ca_mask=ca_mask,
                    edge_mask=edge_mask,
                    batch_pocket=batch_pocket,
                    edge_mask_pocket=edge_mask_pocket,
                    property_tau=property_tau,
                    kind="joint",  # currently hardcoded for max. two properties, whereby SA is always the first and the property the last argument!
                    property_model=property_model,
                    ensemble_models=ckpts_ensemble,
                    property_normalization=property_normalization,
                    edge_attr_initial_ohe=edge_initial_interaction,
                    latent_gamma=latent_gamma,
                )

                atom_types, charge_types = node_feats_in.split(
                    [self.num_atom_types, self.num_charge_classes], dim=-1
                )
                jj, ii = edge_index_global_lig
                mask = jj < ii
                mask_i = ii[mask]

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
            (
                edge_index_global,
                edge_attr_global,
                batch_edge_global,
                edge_mask,
                edge_mask_pocket,
                edge_initial_interaction,
            ) = get_joint_edge_attrs(
                pos,
                pos_pocket,
                batch,
                batch_pocket,
                edge_attr_global_lig,
                self.num_bond_classes,
                self.device,
                cutoff_p=self.cutoff_p,
                cutoff_lp=self.cutoff_lp,
                knn=self.knn,
                hybrid_knn=self.hybrid_knn,
                knn_with_cutoff=self.knn_with_cutoff,
                pocket_mask=pocket_mask,
            )

            if save_traj:
                atom_decoder = self.dataset_info.atom_decoder
                write_xyz_file_from_batch(
                    pos + pocket_cog[batch],
                    atom_types,
                    batch,
                    pos_pocket=pos_pocket + pocket_cog[batch_pocket],
                    atoms_pocket=atom_types_pocket,
                    batch_pocket=batch_pocket,
                    pocket_name=(
                        pocket_data.pocket_name
                        if "pocket_name" in pocket_data
                        else None
                    ),
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
            pocket_name=(
                pocket_data.pocket_name if "pocket_name" in pocket_data else None
            ),
            device=self.device,
            mol_device="cpu",
            context=None,
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

    def select_splitted_node_feats(self, x: Tensor, batch_num_nodes: Tensor, select: Tensor):
        x_split = x.split(batch_num_nodes.cpu().numpy().tolist(), dim=0)
        x_select = torch.concat([x_split[i] for i in select.cpu().numpy()], dim=0)
        return x_select.to(x.device)
    
    def _optimal_transport_alignment(self, a: Tensor, b: Tensor):
        C = torch.cdist(a, b, p=2)
        _, dest_ind = linear_sum_assignment(C.cpu().numpy(), maximize=False)
        dest_ind = torch.tensor(dest_ind, device=a.device)
        b_sorted = b[dest_ind]
        return b_sorted
        
    def optimal_transport_alignment(self, 
                                    pos_ligand: Tensor,
                                    pos_random: Tensor,
                                    batch: Tensor,
                                    ):
        # Performs earth-mover distance optimal transport alignment between batch of two point clouds    
        pos_ligand_splits = pos_ligand.split(batch.bincount().tolist(), dim=0)
        pos_random_splits = pos_random.split(batch.bincount().tolist(), dim=0)
        
        pos_random_updated = [self._optimal_transport_alignment(a, b) for a, b in zip(pos_ligand_splits,
                                                                                      pos_random_splits)
                              ]
        pos_random_updated = torch.cat(pos_random_updated, dim=0)
        return pos_random_updated
        
    def inpainting(
        self,
        num_graphs: int,
        pocket_data: Tensor,
        verbose: bool = False,
        relax_mol=False,
        max_relax_iter=200,
        sanitize=False,
        build_obabel_mol=False,
        clash_guidance: bool = False,
        clash_guidance_start=None,
        clash_guidance_end=None,
        clash_guidance_scale: float = 0.1,
        sa_importance_sampling=False,
        sa_importance_sampling_start=0,
        sa_importance_sampling_end=200,
        sa_every_importance_t=5,
        sa_tau=0.1,
        fix_coords=True,
        emd_ot=False,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, List]:
        
        pos_pocket = pocket_data.pos_pocket.to(self.device)
        pos_ligand = pocket_data.pos.to(self.device)
        batch_pocket = pocket_data.pos_pocket_batch.to(self.device)
        batch_ligand = pocket_data.batch.to(self.device)
        lig_inpaint_mask = pocket_data.lig_inpaint_mask.to(self.device)
        lig_inpaint_mask_f = lig_inpaint_mask.float().unsqueeze(-1)
        
        pos_ligand_initial = pocket_data.pos.to(self.device)
        atom_types_ligand_initial = pocket_data.x.to(self.device)
        edge_index_initial = pocket_data.edge_index.to(self.device)
        x_pocket = pocket_data.x_pocket.to(self.device)

        try:
            ca_mask = pocket_data.ca_mask.to(self.device)
        except Exception:
            ca_mask = None

        bs = num_graphs
                      
        pocket_cog = scatter_mean(pos_pocket, batch_pocket, dim=0)
        pos_pocket = pos_pocket - pocket_cog[batch_pocket]
        pos_ligand_initial = pos_ligand_initial - pocket_cog[batch_ligand]
        
        # initialize random coordinates for all ligand atoms
        pos_ligand = torch.randn_like(pos_ligand_initial)
        # optimal transport alignment to "align" input molecule point cloud with random noise
        if emd_ot:
            pos_ligand = self.optimal_transport_alignment(pos_ligand_initial, pos_ligand, batch_ligand)
            
        n = len(pos_ligand)

        # ligand
        atom_types_ligand = torch.multinomial(
            self.atoms_prior, num_samples=n, replacement=True
        )
        atom_types_ligand = F.one_hot(atom_types_ligand, self.num_atom_types).float()
        charges_ligand = torch.multinomial(
            self.charges_prior, num_samples=n, replacement=True
        )
        charges_ligand = F.one_hot(charges_ligand, self.num_charge_classes).float()
       
       # pocket
        atom_types_pocket = F.one_hot(
            x_pocket.squeeze().long(), num_classes=self.num_atom_types
        ).float()
        charges_pocket = torch.zeros(
            pos_pocket.shape[0], charges_ligand.shape[1], dtype=torch.float32
        ).to(self.device)

        edge_index_local = None
        edge_index_global = (
            torch.eq(batch_ligand.unsqueeze(0), batch_ligand.unsqueeze(-1))
            .int()
            .fill_diagonal_(0)
        )
        edge_index_global, _ = dense_to_sparse(edge_index_global)
        edge_index_global = sort_edge_index(edge_index_global, sort_by_row=False)
        # populate locate edge index into global
        ## TODO
        
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

        (
            pos_joint,
            atom_types_joint,
            charge_types_joint,
            batch_full,
            pocket_mask,
        ) = concat_ligand_pocket(
            pos_ligand,
            pos_pocket,
            atom_types_ligand,
            atom_types_pocket,
            charges_ligand,
            charges_pocket,
            batch_ligand,
            batch_pocket,
            sorting=False,
        )
        (
            edge_index_global,
            edge_attr_global,
            batch_edge_global,
            edge_mask,
            edge_mask_pocket,
            edge_initial_interaction,
        ) = get_joint_edge_attrs(
            pos_ligand,
            pos_pocket,
            batch_ligand,
            batch_pocket,
            edge_attr_global_lig,
            self.num_bond_classes,
            self.device,
            cutoff_p=self.cutoff_p,
            cutoff_lp=self.cutoff_lp,
            knn=self.knn,
            hybrid_knn=self.hybrid_knn,
            knn_with_cutoff=self.knn_with_cutoff,
            pocket_mask=pocket_mask,
        )

        if clash_guidance_start is None:
            clash_guidance_start = 0
        if clash_guidance_end is None:
            clash_guidance_end = self.hparams.timesteps
            
        chain = range(0, self.hparams.timesteps)

        iterator = (
            tqdm(reversed(chain), total=len(chain)) if verbose else reversed(chain)
        )
        
        num_nodes_lig = batch_ligand.bincount().to(self.device)
        
        for i, timestep in enumerate(iterator):
            
            s = torch.full(
                size=(bs,), fill_value=timestep, dtype=torch.long, device=self.device
            )
            t = s + 1
            
            node_feats_in = torch.cat([atom_types_joint, charge_types_joint], dim=-1)
            temb = t / self.hparams.timesteps
            temb = temb.unsqueeze(dim=1)
                    
            out = self.model(
                x=node_feats_in,
                z=None,
                t=temb,
                pos=pos_joint,
                edge_index_local=edge_index_local,
                edge_index_global=edge_index_global,
                edge_index_global_lig=edge_index_global_lig,
                edge_attr_global=edge_attr_global,
                batch=batch_full,
                batch_edge_global=batch_edge_global,
                context=None,
                pocket_mask=pocket_mask.unsqueeze(1),
                edge_mask=edge_mask,
                edge_mask_pocket=edge_mask_pocket,
                batch_lig=batch_ligand,
                ca_mask=ca_mask,
                batch_pocket=batch_pocket,
                edge_attr_initial_ohe=edge_initial_interaction,
                latent_gamma=None,
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
            # positions
            pos_ligand = self.sde_pos.sample_reverse_adaptive(
                s, t, pos_ligand, coords_pred, batch_ligand, cog_proj=False, eta_ddim=1.0
            ) # here is cog_proj false as it will be downprojected later      
            # atoms
            atom_types_ligand = self.cat_atoms.sample_reverse_categorical(
                xt=atom_types_ligand,
                x0=atoms_pred,
                t=t[batch_ligand],
                num_classes=self.num_atom_types,
            )
            
            # inpainting through forward noising
            _, pos_ligand_inpaint = self.sde_pos.sample_pos(
                s,
                pos_ligand_initial,
                batch_ligand,
                remove_mean=False,
            )            
            _, atom_types_inpaint = self.cat_atoms.sample_categorical(
                s,
                atom_types_ligand_initial,
                batch_ligand,
                self.dataset_info,
                num_classes=self.num_atom_types,
                type="atoms",
            )
            # combine
            pos_ligand = pos_ligand * (1.0 - lig_inpaint_mask_f) + pos_ligand_inpaint * lig_inpaint_mask_f
            atom_types_ligand = atom_types_ligand * (1.0 - lig_inpaint_mask_f) + atom_types_inpaint * lig_inpaint_mask_f
            
            # charges
            charges_ligand = self.cat_charges.sample_reverse_categorical(
                xt=charges_ligand,
                x0=charges_pred,
                t=t[batch_ligand],
                num_classes=self.num_charge_classes,
            )
            # edges
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
                batch=batch_ligand,
                edge_index_global=edge_index_global_lig,
                num_classes=self.num_bond_classes,
            )
            
            if clash_guidance:
                if clash_guidance_start <= i <= clash_guidance_end:
                    _, delta = pocket_clash_guidance(x_l=pos_ligand, x_p=pos_pocket,
                                                    batch_l=batch_ligand, batch_p=batch_pocket,
                                                    sigma=2.0
                                                    )
                    pos_ligand = pos_ligand + clash_guidance_scale * delta
                    
            # concatenate again into joint ligand-pocket graph
            (
            pos_joint,
            atom_types_joint,
            charge_types_joint,
            batch_full,
            pocket_mask,
            ) = concat_ligand_pocket(
            pos_ligand,
            pos_pocket,
            atom_types_ligand,
            atom_types_pocket,
            charges_ligand,
            charges_pocket,
            batch_ligand,
            batch_pocket,
            sorting=False,
            )
            
            (
            edge_index_global,
            edge_attr_global,
            batch_edge_global,
            edge_mask,
            edge_mask_pocket,
            edge_initial_interaction,
            ) = get_joint_edge_attrs(
            pos_ligand,
            pos_pocket,
            batch_ligand,
            batch_pocket,
            edge_attr_global_lig,
            self.num_bond_classes,
            self.device,
            cutoff_p=self.cutoff_p,
            cutoff_lp=self.cutoff_lp,
            knn=self.knn,
            hybrid_knn=self.hybrid_knn,
            knn_with_cutoff=self.knn_with_cutoff,
            pocket_mask=pocket_mask,
            )
            
            # sa importance sampling
            if sa_importance_sampling and i % sa_every_importance_t == 0 and sa_importance_sampling_start <= i <= sa_importance_sampling_end:
                
                node_feats_in = torch.cat(
                    [atom_types_joint, charge_types_joint], dim=-1
                )
                (
                    pos_ligand,
                    node_feats_in,
                    edge_index_global_lig,
                    edge_attr_global_lig,
                    batch_ligand,
                    _tmp,
                    num_nodes_lig,
                ) = self.importance_sampling(
                    node_feats_in=node_feats_in,
                    pos=pos_joint,
                    temb=temb,
                    z=None,
                    edge_index_local=None,
                    edge_index_global=edge_index_global,
                    edge_attr_global=edge_attr_global,
                    batch=batch_full,
                    batch_lig=batch_ligand,
                    batch_edge_global=batch_edge_global,
                    batch_num_nodes=num_nodes_lig,
                    context=None,
                    sa_tau=sa_tau,
                    maximize_score=True,
                    edge_index_global_lig=edge_index_global_lig,
                    edge_attr_global_lig=edge_attr_global_lig,
                    pocket_mask=pocket_mask,
                    ca_mask=ca_mask,
                    edge_mask=edge_mask,
                    batch_pocket=batch_pocket,
                    edge_mask_pocket=edge_mask_pocket,
                    kind="sa_score",
                    sa_model=None,
                    ensemble_models=[],
                    property_normalization=False,
                    edge_attr_initial_ohe=edge_initial_interaction,
                    latent_gamma=None,
                )
                
                atom_types_ligand, charges_ligand = node_feats_in.split(
                    [self.num_atom_types, self.num_charge_classes], dim=-1
                )
                jj, ii = edge_index_global_lig
                mask = jj < ii
                mask_i = ii[mask]
                
                # select initials
                pos_ligand_initial = self.select_splitted_node_feats(pos_ligand_initial, _tmp.get("batch_num_nodes_old"), _tmp.get("select"))
                atom_types_ligand_initial = self.select_splitted_node_feats(atom_types_ligand_initial, _tmp.get("batch_num_nodes_old"), _tmp.get("select"))
                lig_inpaint_mask = self.select_splitted_node_feats(lig_inpaint_mask, _tmp.get("batch_num_nodes_old"), _tmp.get("select"))
                lig_inpaint_mask_f = self.select_splitted_node_feats(lig_inpaint_mask_f, _tmp.get("batch_num_nodes_old"), _tmp.get("select"))
                
                # inpainting through forward noising
                _, pos_ligand_inpaint = self.sde_pos.sample_pos(
                    s,
                    pos_ligand_initial,
                    batch_ligand,
                    remove_mean=False,
                )            
                _, atom_types_inpaint = self.cat_atoms.sample_categorical(
                    s,
                    atom_types_ligand_initial,
                    batch_ligand,
                    self.dataset_info,
                    num_classes=self.num_atom_types,
                    type="atoms",
                )
                # combine
                pos_ligand = pos_ligand * (1.0 - lig_inpaint_mask_f) + pos_ligand_inpaint * lig_inpaint_mask_f
                atom_types_ligand = atom_types_ligand * (1.0 - lig_inpaint_mask_f) + atom_types_inpaint * lig_inpaint_mask_f
                # concatenate again into joint ligand-pocket graph
            (
                pos_joint,
                atom_types_joint,
                charge_types_joint,
                batch_full,
                pocket_mask,
            ) = concat_ligand_pocket(
                pos_ligand,
                pos_pocket,
                atom_types_ligand,
                atom_types_pocket,
                charges_ligand,
                charges_pocket,
                batch_ligand,
                batch_pocket,
                sorting=False,
                )
            (
                edge_index_global,
                edge_attr_global,
                batch_edge_global,
                edge_mask,
                edge_mask_pocket,
                edge_initial_interaction,
            ) = get_joint_edge_attrs(
                pos_ligand,
                pos_pocket,
                batch_ligand,
                batch_pocket,
                edge_attr_global_lig,
                self.num_bond_classes,
                self.device,
                cutoff_p=self.cutoff_p,
                cutoff_lp=self.cutoff_lp,
                knn=self.knn,
                hybrid_knn=self.hybrid_knn,
                knn_with_cutoff=self.knn_with_cutoff,
                pocket_mask=pocket_mask,
            )
            
        # at last step, just infill the ground truth inpaintings
        pos_ligand = pos_ligand * (1.0 - lig_inpaint_mask_f) + pos_ligand_initial * lig_inpaint_mask_f
        atom_types_ligand_initial = F.one_hot(atom_types_ligand_initial, atom_types_inpaint.size(1)).float()
        atom_types_ligand = atom_types_ligand * (1.0 - lig_inpaint_mask_f) + atom_types_ligand_initial * lig_inpaint_mask_f

        # Move generated molecule back to the original pocket position for docking
        pos_ligand += pocket_cog[batch_ligand]
        pos_pocket += pocket_cog[batch_pocket]

        out_dict = {
            "coords_pred": pos_ligand,
            "coords_pocket": pos_pocket,
            "atoms_pred": atom_types_ligand,
            "atoms_pocket": atom_types_pocket,
            "charges_pred": charges_ligand,
            "bonds_pred": edge_attr_global_lig,
        }
        molecules = get_molecules(
            out_dict,
            batch_ligand,
            edge_index_global_lig,
            self.num_atom_types,
            self.num_charge_classes,
            self.dataset_info,
            data_batch_pocket=batch_pocket,
            pocket_name=(
                pocket_data.pocket_name if "pocket_name" in pocket_data else None
            ),
            device=self.device,
            mol_device="cpu",
            context=None,
            relax_mol=relax_mol,
            max_relax_iter=max_relax_iter,
            sanitize=sanitize,
            while_train=False,
            build_obabel_mol=build_obabel_mol,
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
        elif self.hparams["lr_scheduler"] == "exponential":
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=0.997
            )
        elif self.hparams["lr_scheduler"] == "cosine_annealing_warmup":
            lr_scheduler = CosineAnnealingWarmupRestarts(
                optimizer,
                first_cycle_steps=100,
                cycle_mult=1.0,
                max_lr=self.hparams.lr,
                min_lr=self.hparams.lr_min,
                warmup_steps=10,
                gamma=0.8,
            )
        else:
            raise Exception("Scheduler not found")
        scheduler = {
            "scheduler": lr_scheduler,
            "interval": "epoch",
            "frequency": self.hparams["lr_frequency"],
            "monitor": self.validity,
            "strict": False,
        }
        return [optimizer], [scheduler]
