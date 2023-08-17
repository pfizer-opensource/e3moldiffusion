import logging
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Batch
from torch_geometric.utils import dense_to_sparse, sort_edge_index, dropout_node

from e3moldiffusion.coordsatomsbonds import DenoisingEdgeNetwork
from e3moldiffusion.molfeat import get_bond_feature_dims
from experiments.diffusion.continuous import DiscreteDDPM
from experiments.diffusion.categorical import CategoricalDiffusionKernel
from experiments.data.abstract_dataset import AbstractDatasetInfos

from experiments.utils import (
    coalesce_edges,
    zero_mean,
)
from experiments.losses import DiffusionLoss

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
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.i = 0

        self.dataset_info = dataset_info

        self.num_atom_types_geom = 16
        atom_types_distribution = dataset_info.atom_types.float()
        if self.hparams.dataset == "pubchem":
            pubchem_ids = [0, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13]
            geom_only = [
                i for i in range(self.num_atom_types_geom) if i not in pubchem_ids
            ]
        atom_types_distribution[geom_only] = 0.0
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

        if hparams.get("no_h"):
            print("Training without hydrogen")
            self.hparams.num_atom_types -= 1

        self.smiles_list = smiles_list

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
            modalities=["coords", "atoms", "charges", "bonds"]
        )

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
            [self.num_atom_types_geom, self.num_charge_classes], dim=-1
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
            weights=None,
        )

        final_loss = (
            3.0 * loss["coords"]
            + 0.4 * loss["atoms"]
            + 2.0 * loss["bonds"]
            + 1.0 * loss["charges"]
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
        n = batch.num_nodes
        bs = int(data_batch.max()) + 1

        bond_edge_index, bond_edge_attr = sort_edge_index(
            edge_index=bond_edge_index, edge_attr=bond_edge_attr, sort_by_row=False
        )

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

        if not self.train:
            # do assertion when valdating
            edge_attr_global_dense_perturbed = torch.zeros(
                n, n, device=pos.device, dtype=torch.long
            )
            edge_attr_global_dense_perturbed[
                edge_index_global_perturbed[0], edge_index_global_perturbed[1]
            ] = edge_attr_global_perturbed
            assert (
                edge_attr_global_dense_perturbed - edge_attr_global_dense_perturbed.T
            ).float().mean().item() == 0.0
            assert torch.allclose(edge_index_global, edge_index_global_perturbed)

        edge_attr_global_perturbed = F.one_hot(
            edge_attr_global_perturbed, num_classes=self.num_bond_classes
        ).float()

        temb = t.float() / self.hparams.timesteps
        temb = temb.clamp(min=self.hparams.eps_min)
        temb = temb.unsqueeze(dim=1)

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

        # one-hot-encode
        if self.hparams.no_h:
            raise NotImplementedError
            node_feat -= 1

        # one-hot-encode atom types
        # atom_types = torch.tensor(
        #     [self.dataset_info.atom_idx_mapping[int(atom)] for atom in atom_types],
        #     device="cuda",
        # ).long()
        atom_types = F.one_hot(
            atom_types.squeeze().long(), num_classes=self.num_atom_types
        ).float()
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

        # MASKING PREDICTION
        edge_index_global, edge_mask, node_mask = dropout_node(
            edge_index_global, p=0.50
        )
        edge_attr_global_perturbed = edge_attr_global_perturbed[edge_mask]
        pos_perturbed = pos_perturbed[node_mask]
        atom_types_perturbed = atom_types_perturbed[node_mask]
        charges_perturbed = charges_perturbed[node_mask]

        batch_edge_global = data_batch[edge_index_global[0]]
        data_batch = data_batch[node_mask]

        atom_feats_in_perturbed = torch.cat(
            [atom_types_perturbed, charges_perturbed], dim=-1
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
        out = self.model(
            x=atom_feats_in_perturbed,
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

        # MASKING LABELS
        edge_attr_global = edge_attr_global[edge_mask]
        pos_centered = pos_centered[node_mask]
        atom_types = atom_types[node_mask]
        charges = charges[node_mask]

        out["coords_true"] = pos_centered
        out["atoms_true"] = atom_types.argmax(dim=-1)
        out["bonds_true"] = edge_attr_global
        out["charges_true"] = charges.argmax(dim=-1)

        out["bond_aggregation_index"] = edge_index_global[1]

        return out

    def _log(
        self, loss, coords_loss, atoms_loss, charges_loss, bonds_loss, batch_size, stage
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
