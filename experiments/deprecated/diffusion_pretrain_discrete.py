import logging

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Batch
from torch_geometric.utils import dense_to_sparse, sort_edge_index

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
from experiments.diffusion.categorical import CategoricalDiffusionKernel
from experiments.diffusion.continuous import DiscreteDDPM
from experiments.losses import DiffusionLoss
from experiments.sampling.utils import calculate_sa
from experiments.utils import (
    coalesce_edges,
    get_lipinski_properties,
    load_latent_encoder,
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

        self.save_hyperparameters(hparams)

        self.cutoff_p = hparams["cutoff_local"]
        self.cutoff_lp = hparams["cutoff_local"]

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

        if self.hparams.ligand_pocket_distance_loss:
            self.dist_loss = torch.nn.HuberLoss(reduction="none", delta=1.0)
        else:
            self.dist_loss = None

    def training_step(self, batch, batch_idx):
        return self.step_fnc(batch=batch, batch_idx=batch_idx, stage="train")

    def validation_step(self, batch, batch_idx):
        return self.step_fnc(batch=batch, batch_idx=batch_idx, stage="val")

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
                or "docking_score" in self.hparams.regression_property
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
                or "ic50" in self.hparams.regression_property
            ):
                if "docking_score" in self.hparams.regression_property:
                    label_prop = batch.docking_scores.float()
                elif "ic50" in self.hparams.regression_property:
                    label_prop = batch.ic50.float()
                else:
                    raise Exception(
                        "Specified regression property ot supported. Choose docking_score or ic50"
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
            batch=batch.batch,
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
            batch_size,
            stage,
        )

        return final_loss

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

        bond_edge_index, bond_edge_attr = sort_edge_index(
            edge_index=bond_edge_index, edge_attr=bond_edge_attr, sort_by_row=False
        )

        # TIME EMBEDDING
        temb = t.float() / self.hparams.timesteps
        temb = temb.clamp(min=self.hparams.eps_min)
        temb = temb.unsqueeze(dim=1)

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

        if self.hparams.model_global_edge and self.hparams.model_edge_rbf_interaction:
            # Get Placeholder for ligand-pocket and pocket-pocket edge interactions for target-aware downstream applications
            edge_mask = (edge_index_global[0] < len(batch)) & (
                edge_index_global[1] < len(batch)
            )
            edge_initial_interaction = torch.zeros(
                (edge_index_global.size(1), 3),
                dtype=torch.float32,
                device=self.device,
            )
            edge_initial_interaction[edge_mask] = (
                torch.tensor([1, 0, 0]).float().to(self.device)
            )  # ligand-ligand
        else:
            edge_initial_interaction = None

        out = self.model(
            x=atom_feats_in_perturbed,
            t=temb,
            pos=pos_perturbed,
            edge_index_local=None,
            edge_index_global=edge_index_global,
            edge_attr_global=(
                edge_attr_global_perturbed if not self.hparams.bond_prediction else None
            ),
            batch=data_batch,
            batch_edge_global=batch_edge_global,
            edge_attr_initial_ohe=edge_initial_interaction,
            context=context,
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

        return out

    def _log(
        self,
        loss,
        coords_loss,
        atoms_loss,
        charges_loss,
        bonds_loss,
        sa_loss,
        property_loss,
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

        if property_loss is not None:
            self.log(
                f"{stage}/property_loss",
                property_loss,
                on_step=True,
                batch_size=batch_size,
                prog_bar=(stage == "train"),
                sync_dist=self.hparams.gpus > 1 and stage == "val",
            )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.hparams["lr"],
            amsgrad=True,
            weight_decay=1e-12,
        )
        # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #    optimizer=optimizer,
        #    patience=self.hparams["lr_patience"],
        #    cooldown=self.hparams["lr_cooldown"],
        #    factor=self.hparams["lr_factor"],
        # )
        # scheduler = {
        #    "scheduler": lr_scheduler,
        #    "interval": "epoch",
        #    "frequency": self.hparams["lr_frequency"],
        #    "monitor": "val/loss",
        #    "strict": False,
        # }
        return [optimizer]  # , [scheduler]
