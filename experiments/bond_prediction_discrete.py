import logging
import os
from datetime import datetime
from typing import List, Tuple

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Batch
from torch_geometric.nn import radius_graph
from torch_geometric.utils import dense_to_sparse, sort_edge_index, dropout_node
from tqdm import tqdm

from e3moldiffusion.coordsatomsbonds import EdgePredictionNetwork
from e3moldiffusion.molfeat import get_bond_feature_dims
from experiments.diffusion.continuous import DiscreteDDPM
from experiments.diffusion.categorical import CategoricalDiffusionKernel

from experiments.molecule_utils import Molecule
from experiments.utils import (
    coalesce_edges,
    get_empirical_num_nodes,
    zero_mean,
)
from experiments.sampling.analyze import analyze_stability_for_molecules
from experiments.losses import PredictionLoss

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
        dataset_info: dict,
        smiles_list: list,
        dataset_statistics=None,
    ):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.i = 0

        self.dataset_statistics = dataset_statistics

        atom_types_distribution = dataset_statistics.atom_types.float()
        bond_types_distribution = dataset_statistics.edge_types.float()
        charge_types_distribution = dataset_statistics.charges_marginals.float()

        self.register_buffer("atoms_prior", atom_types_distribution.clone())
        self.register_buffer("bonds_prior", bond_types_distribution.clone())
        self.register_buffer("charges_prior", charge_types_distribution.clone())

        self.hparams.num_atom_types = dataset_statistics.input_dims.X
        self.num_charge_classes = dataset_statistics.input_dims.C
        self.num_atom_types = self.hparams.num_atom_types
        self.num_atom_features = self.num_atom_types + self.num_charge_classes
        self.num_bond_classes = 5

        if hparams.get("no_h"):
            print("Training without hydrogen")
            self.hparams.num_atom_types -= 1

        self.smiles_list = smiles_list

        self.dataset_info = dataset_info

        empirical_num_nodes = get_empirical_num_nodes(dataset_info)
        self.register_buffer(name="empirical_num_nodes", tensor=empirical_num_nodes)

        self.model = EdgePredictionNetwork(
            hn_dim=(hparams["sdim"], hparams["vdim"]),
            num_layers=hparams["num_layers"],
            use_norm=not hparams["omit_norm"],
            use_cross_product=not hparams["omit_cross_product"],
            num_atom_features=self.num_atom_features,
            num_bond_types=self.num_bond_classes,
            edge_dim=hparams["edim"],
            cutoff_local=hparams["cutoff_local"],
            rbf_dim=hparams["rbf_dim"],
            vector_aggr=hparams["vector_aggr"],
            fully_connected=hparams["fully_connected"],
            local_global_model=hparams["local_global_model"],
            recompute_edge_attributes=True,
            recompute_radius_graph=False,
        )

        self.sde = DiscreteDDPM(
            beta_min=hparams["beta_min"],
            beta_max=hparams["beta_max"],
            N=hparams["timesteps"],
            scaled_reverse_posterior_sigma=True,
            schedule="cosine",
            enforce_zero_terminal_snr=False,
        )

        self.cat_atoms = CategoricalDiffusionKernel(
            terminal_distribution=atom_types_distribution,
            alphas=self.sde.alphas.clone(),
        )
        self.cat_bonds = CategoricalDiffusionKernel(
            terminal_distribution=bond_types_distribution,
            alphas=self.sde.alphas.clone(),
        )
        self.cat_charges = CategoricalDiffusionKernel(
            terminal_distribution=charge_types_distribution,
            alphas=self.sde.alphas.clone(),
        )

        self.prediction_loss = PredictionLoss()

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

        edges_true = {
            "bonds": out_dict["bonds_true"],
        }
        edges_pred = out_dict["bonds_pred"]

        loss = self.prediction_loss(
            true_data=edges_true,
            pred_data=edges_pred,
        )

        return loss

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
        mean_coords, std_coords = self.sde.marginal_prob(
            x=pos_centered, t=t[data_batch]
        )
        # perturb coords
        pos_perturbed = mean_coords + std_coords * noise_coords_true

        # one-hot-encode
        if self.hparams.no_h:
            raise NotImplementedError
            node_feat -= 1

        # one-hot-encode atom types
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
        charges = self.dataset_statistics.one_hot_charges(charges)
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

        return out

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
