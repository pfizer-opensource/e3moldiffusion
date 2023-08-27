import logging
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from e3moldiffusion.coordsatomsbonds import DenoisingEdgeNetwork
from e3moldiffusion.molfeat import get_bond_feature_dims
from experiments.data.abstract_dataset import AbstractDatasetInfos
from experiments.data.distributions import prepare_context
from torch_geometric.utils import dense_to_sparse, sort_edge_index, dropout_node
from experiments.diffusion.categorical import CategoricalDiffusionKernel
from experiments.diffusion.continuous import DiscreteDDPM
from experiments.losses import DiffusionLoss
from experiments.utils import (
    coalesce_edges,
    load_model,
    zero_mean,
)
from torch import Tensor
from torch_geometric.data import Batch

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
    ):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.i = 0

        self.dataset_info = dataset_info
        self.prop_norm = prop_norm
        self.prop_dist = prop_dist

        self.num_atom_types_geom = 16
        if self.hparams.dataset == "pubchem":
            pubchem_ids = [0, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13]
            geom_only = [
                i for i in range(self.num_atom_types_geom) if i not in pubchem_ids
            ]
        atom_types_distribution = dataset_info.atom_types.float()
        atom_types_distribution[geom_only] = 0.0
        bond_types_distribution = dataset_info.edge_types.float()
        charge_types_distribution = dataset_info.charges_marginals.float()
        is_aromatic_distribution = dataset_info.is_aromatic.float()
        is_ring_distribution = dataset_info.is_in_ring.float()
        hybridization_disitribution = dataset_info.hybridization.float()

        self.register_buffer("atoms_prior", atom_types_distribution.clone())
        self.register_buffer("bonds_prior", bond_types_distribution.clone())
        self.register_buffer("charges_prior", charge_types_distribution.clone())
        self.register_buffer("is_aromatic_prior", is_aromatic_distribution.clone())
        self.register_buffer("is_in_ring_prior", is_ring_distribution.clone())
        self.register_buffer("hybridization_prior", hybridization_disitribution.clone())

        self.num_is_aromatic = self.num_is_in_ring = 2
        self.num_hybridization = 9

        self.hparams.num_atom_types = dataset_info.input_dims.X
        self.num_charge_classes = dataset_info.input_dims.C
        self.num_atom_types = self.hparams.num_atom_types
        self.num_atom_features = (
            self.num_atom_types
            + self.num_charge_classes
            + self.num_is_aromatic
            + self.num_is_in_ring
            + self.num_hybridization
        )
        self.num_bond_classes = 5
        self.remove_hs = hparams.get("remove_hs")
        if self.remove_hs:
            print("Model without modelling explicit hydrogens")
        self.smiles_list = smiles_list

        empirical_num_nodes = dataset_info.n_nodes
        self.register_buffer(name="empirical_num_nodes", tensor=empirical_num_nodes)

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
            bond_prediction=hparams["bond_prediction"],
            property_prediction=hparams["property_prediction"],
            coords_param=hparams["continuous_param"],
        )

        self.sde_pos = DiscreteDDPM(
            beta_min=hparams["beta_min"],
            beta_max=hparams["beta_max"],
            N=hparams["timesteps"],
            scaled_reverse_posterior_sigma=True,
            schedule=self.hparams.noise_scheduler,
            nu=2.5,
            enforce_zero_terminal_snr=False,
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
        )
        self.cat_bonds = CategoricalDiffusionKernel(
            terminal_distribution=bond_types_distribution,
            alphas=self.sde_bonds.alphas.clone(),
        )
        self.cat_charges = CategoricalDiffusionKernel(
            terminal_distribution=charge_types_distribution,
            alphas=self.sde_atom_charge.alphas.clone(),
        )
        self.cat_aromatic = CategoricalDiffusionKernel(
            terminal_distribution=is_aromatic_distribution,
            alphas=self.sde_atom_charge.alphas.clone(),
        )
        self.cat_ring = CategoricalDiffusionKernel(
            terminal_distribution=is_ring_distribution,
            alphas=self.sde_atom_charge.alphas.clone(),
        )
        self.cat_hybridization = CategoricalDiffusionKernel(
            terminal_distribution=hybridization_disitribution,
            alphas=self.sde_atom_charge.alphas.clone(),
        )

        self.diffusion_loss = DiffusionLoss(
            modalities=[
                "coords",
                "atoms",
                "charges",
                "bonds",
                "ring",
                "aromatic",
                "hybridization",
            ],
            param=["data"] * 7,
        )

    def training_step(self, batch, batch_idx):
        return self.step_fnc(batch=batch, batch_idx=batch_idx, stage="train")

    def validation_step(self, batch, batch_idx):
        return self.step_fnc(batch=batch, batch_idx=batch_idx, stage="val")

    def _log(
        self,
        loss,
        coords_loss,
        atoms_loss,
        charges_loss,
        bonds_loss,
        ring_loss,
        aromatic_loss,
        hybridization_loss,
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
            f"{stage}/ring_loss",
            ring_loss,
            on_step=True,
            batch_size=batch_size,
            prog_bar=(stage == "train"),
            sync_dist=self.hparams.gpus > 1 and stage == "val",
        )
        self.log(
            f"{stage}/aromatic_loss",
            aromatic_loss,
            on_step=True,
            batch_size=batch_size,
            prog_bar=(stage == "train"),
            sync_dist=self.hparams.gpus > 1 and stage == "val",
        )
        self.log(
            f"{stage}/hybridization_loss",
            hybridization_loss,
            on_step=True,
            batch_size=batch_size,
            prog_bar=(stage == "train"),
            sync_dist=self.hparams.gpus > 1 and stage == "val",
        )

    def step_fnc(self, batch, batch_idx, stage: str):
        if self.hparams.context_mapping:
            context = prepare_context(
                self.hparams["properties_list"],
                self.prop_norm,
                batch,
                self.hparams.dataset,
            )
            batch.context = context

        out_dict, t, node_mask, batch_size = self(batch=batch)
        data_batch = batch.batch[node_mask]

        if self.hparams.loss_weighting == "snr_s_t":
            weights = self.sde_atom_charge.snr_s_t_weighting(
                s=t - 1, t=t, clamp_min=None, clamp_max=None
            ).to(batch.x.device)
        elif self.hparams.loss_weighting == "snr_t":
            weights = self.sde_atom_charge.snr_t_weighting(
                t=t, device=batch.x.device, clamp_min=0.05, clamp_max=5.0
            )
        elif self.hparams.loss_weighting == "exp_t":
            weights = self.sde_atom_charge.exp_t_weighting(t=t, device=batch.x.device)
        elif self.hparams.loss_weighting == "exp_t_half":
            weights = self.sde_atom_charge.exp_t_half_weighting(
                t=t, device=batch.x.device
            )
        elif self.hparams.loss_weighting == "uniform":
            weights = None
        true_data = {
            "coords": out_dict["coords_true"]
            if self.hparams.continuous_param == "data"
            else out_dict["coords_noise_true"],
            "atoms": out_dict["atoms_true"],
            "charges": out_dict["charges_true"],
            "bonds": out_dict["bonds_true"],
            "ring": out_dict["ring_true"],
            "aromatic": out_dict["aromatic_true"],
            "hybridization": out_dict["hybridization_true"],
        }

        coords_pred = out_dict["coords_pred"]
        atoms_pred = out_dict["atoms_pred"]
        (
            atoms_pred,
            charges_pred,
            ring_pred,
            aromatic_pred,
            hybridization_pred,
        ) = atoms_pred.split(
            [
                self.num_atom_types,
                self.num_charge_classes,
                self.num_is_in_ring,
                self.num_is_aromatic,
                self.num_hybridization,
            ],
            dim=-1,
        )
        edges_pred = out_dict["bonds_pred"]

        pred_data = {
            "coords": coords_pred,
            "atoms": atoms_pred,
            "charges": charges_pred,
            "bonds": edges_pred,
            "ring": ring_pred,
            "aromatic": aromatic_pred,
            "hybridization": hybridization_pred,
        }

        loss = self.diffusion_loss(
            true_data=true_data,
            pred_data=pred_data,
            batch=data_batch,
            bond_aggregation_index=out_dict["bond_aggregation_index"],
            weights=weights,
        )

        final_loss = (
            self.hparams.lc_coords * loss["coords"]
            + self.hparams.lc_atoms * loss["atoms"]
            + self.hparams.lc_bonds * loss["bonds"]
            + self.hparams.lc_charges * loss["charges"]
            + 0.5 * loss["ring"]
            + 0.7 * loss["aromatic"]
            + 1.0 * loss["hybridization"]
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
            loss["ring"],
            loss["aromatic"],
            loss["hybridization"],
            batch_size,
            stage,
        )

        return final_loss

    def forward(self, batch: Batch):
        atom_types: Tensor = batch.x
        pos: Tensor = batch.pos
        charges: Tensor = batch.charges
        data_batch: Tensor = batch.batch
        bond_edge_index = batch.edge_index
        bond_edge_attr = batch.edge_attr
        context = batch.context if self.hparams.context_mapping else None
        n = batch.num_nodes
        bs = int(data_batch.max()) + 1
        t = torch.randint(
            low=1,
            high=self.hparams.timesteps + 1,
            size=(bs,),
            dtype=torch.long,
            device=batch.x.device,
        )

        ring_feat = batch.is_in_ring
        aromatic_feat = batch.is_aromatic
        hybridization_feat = batch.hybridization

        bond_edge_index, bond_edge_attr = sort_edge_index(
            edge_index=bond_edge_index, edge_attr=bond_edge_attr, sort_by_row=False
        )

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
            t, atom_types, data_batch, self.dataset_info, type="atoms"
        )
        charges, charges_perturbed = self.cat_charges.sample_categorical(
            t, charges, data_batch, self.dataset_info, type="charges"
        )

        edge_attr_global_perturbed = (
            self.cat_bonds.sample_edges_categorical(
                t, edge_index_global, edge_attr_global, data_batch
            )
            if not self.hparams.bond_prediction
            else None
        )

        ## tempory: use this version without refactored.
        # ring-feat and perturb
        ring_feat = F.one_hot(
            ring_feat.squeeze().long(), num_classes=self.num_is_in_ring
        ).float()
        probs = self.cat_ring.marginal_prob(ring_feat, t[data_batch])
        ring_feat_perturbed = F.one_hot(
            probs.multinomial(
                1,
            ).squeeze(),
            num_classes=self.num_is_in_ring,
        ).float()

        # aromatic-feat and perturb
        aromatic_feat = F.one_hot(
            aromatic_feat.squeeze().long(), num_classes=self.num_is_aromatic
        ).float()
        probs = self.cat_aromatic.marginal_prob(aromatic_feat, t[data_batch])
        aromatic_feat_perturbed = F.one_hot(
            probs.multinomial(
                1,
            ).squeeze(),
            num_classes=self.num_is_aromatic,
        ).float()

        # hybridization and perturb
        hybridization_feat = F.one_hot(
            hybridization_feat.squeeze().long(), num_classes=self.num_hybridization
        ).float()
        probs = self.cat_hybridization.marginal_prob(hybridization_feat, t[data_batch])
        hybridization_feat_perturbed = F.one_hot(
            probs.multinomial(
                1,
            ).squeeze(),
            num_classes=self.num_hybridization,
        ).float()

        # MASKING PREDICTION
        edge_index_global, edge_mask, node_mask = dropout_node(
            edge_index_global,
            p=self.hparams.dropout_prob,
        )
        edge_attr_global_perturbed = edge_attr_global_perturbed[edge_mask]
        pos_perturbed = pos_perturbed[node_mask]
        atom_types_perturbed = atom_types_perturbed[node_mask]
        charges_perturbed = charges_perturbed[node_mask]
        ring_feat_perturbed = ring_feat_perturbed[node_mask]
        aromatic_feat_perturbed = aromatic_feat_perturbed[node_mask]
        hybridization_feat_perturbed = hybridization_feat_perturbed[node_mask]

        batch_edge_global = data_batch[edge_index_global[0]]
        data_batch = data_batch[node_mask]
        batch_size = len(data_batch.unique())

        atom_feats_in_perturbed = torch.cat(
            [
                atom_types_perturbed,
                charges_perturbed,
                ring_feat_perturbed,
                aromatic_feat_perturbed,
                hybridization_feat_perturbed,
            ],
            dim=-1,
        )

        edge_index_global = (
            torch.eq(data_batch.unsqueeze(0), data_batch.unsqueeze(-1))
            .int()
            .fill_diagonal_(0)
        )
        edge_index_global, _ = dense_to_sparse(edge_index_global)
        edge_index_global, edge_attr_global_perturbed = sort_edge_index(
            edge_index=edge_index_global,
            edge_attr=edge_attr_global_perturbed,
            sort_by_row=False,
        )

        # TIME EMBEDDING
        t = t[data_batch.unique()]
        temb = t.float() / self.hparams.timesteps
        temb = temb.clamp(min=self.hparams.eps_min)
        temb = temb.unsqueeze(dim=1)

        # FORWARD
        out = self.model(
            x=atom_feats_in_perturbed,
            t=temb,
            pos=pos_perturbed,
            edge_index_local=None,
            edge_index_global=edge_index_global,
            edge_attr_global=edge_attr_global_perturbed,
            batch=data_batch,
            batch_edge_global=batch_edge_global,
            context=context,
        )

        out["coords_perturbed"] = pos_perturbed
        out["atoms_perturbed"] = atom_types_perturbed
        out["charges_perturbed"] = charges_perturbed
        out["bonds_perturbed"] = edge_attr_global_perturbed
        out["ring_perturbed"] = ring_feat_perturbed
        out["aromatic_perturbed"] = aromatic_feat_perturbed
        out["hybridization_perturbed"] = hybridization_feat_perturbed

        # MASKING LABELS
        edge_attr_global = edge_attr_global[edge_mask]
        pos_centered = pos_centered[node_mask]
        atom_types = atom_types[node_mask]
        charges = charges[node_mask]
        ring_feat = ring_feat[node_mask]
        aromatic_feat = aromatic_feat[node_mask]
        hybridization_feat = hybridization_feat[node_mask]

        out["coords_true"] = pos_centered
        out["coords_noise_true"] = noise_coords_true
        out["atoms_true"] = atom_types.argmax(dim=-1)
        out["bonds_true"] = edge_attr_global
        out["charges_true"] = charges.argmax(dim=-1)
        out["ring_true"] = ring_feat.argmax(dim=-1)
        out["aromatic_true"] = aromatic_feat.argmax(dim=-1)
        out["hybridization_true"] = hybridization_feat.argmax(dim=-1)

        out["bond_aggregation_index"] = edge_index_global[1]

        return out, t, node_mask, batch_size

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.hparams["lr"],
            amsgrad=True,
            weight_decay=1.0e-12,
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
