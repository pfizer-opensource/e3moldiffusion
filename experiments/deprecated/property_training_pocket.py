import logging

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Batch
from torch_geometric.utils import dense_to_sparse, sort_edge_index

from e3moldiffusion.coordsatomsbonds import EQGATEnergyNetwork, PropertyEdgeNetwork
from e3moldiffusion.molfeat import get_bond_feature_dims
from experiments.data.abstract_dataset import AbstractDatasetInfos
from experiments.data.distributions import PROP_TO_IDX_GEOMQM as prop_to_idx
from experiments.data.distributions import prepare_context
from experiments.diffusion.categorical import CategoricalDiffusionKernel
from experiments.diffusion.continuous import DiscreteDDPM
from experiments.diffusion.utils import (
    get_joint_edge_attrs,
)
from experiments.sampling.utils import calculate_sascore
from experiments.utils import (
    coalesce_edges,
    concat_ligand_pocket,
    remove_mean_pocket,
)

logging.getLogger("lightning").setLevel(logging.WARNING)

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

        self.dataset_info = dataset_info
        self.prop_norm = prop_norm
        self.prop_dist = prop_dist

        # self.property_idx = prop_to_idx[self.hparams.regression_property]

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
            print("Model without modelling explicit hydrogens")

        self.num_atom_types = self.hparams.num_atom_types
        self.num_atom_features = self.num_atom_types + self.num_charge_classes
        self.num_bond_classes = 5

        self.smiles_list = smiles_list

        empirical_num_nodes = dataset_info.n_nodes
        self.register_buffer(name="empirical_num_nodes", tensor=empirical_num_nodes)

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

        self.model = PropertyEdgeNetwork(
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
            ligand_pocket_interaction=hparams["ligand_pocket_interaction"],
            bond_prediction=hparams["bond_prediction"],
            property_prediction=hparams["property_prediction"],
            joint_property_prediction=hparams["joint_property_prediction"],
        )

        # self.model = EQGATEnergyNetwork(
        #     hn_dim=(hparams["sdim"], hparams["vdim"]),
        #     num_layers=hparams["num_layers"],
        #     num_rbfs=hparams["rbf_dim"],
        #     use_cross_product=hparams["use_cross_product"],
        #     num_atom_features=self.num_atom_features,
        #     cutoff_local=hparams["cutoff_local"],
        #     vector_aggr=hparams["vector_aggr"],
        # )

        if self.hparams.regression_property == "docking_score":
            self.loss = torch.nn.MSELoss(reduce=False, reduction="none")
        elif self.hparams.regression_property == "sascore":
            self.loss = torch.nn.BCEWithLogitsLoss(reduce=False, reduction="none")

        # initialize loss collection
        self.losses = None
        self._reset_losses_dict()

    def training_step(self, batch, batch_idx):
        return self.step_fnc(batch=batch, batch_idx=batch_idx, stage="train")

    def validation_step(self, batch, batch_idx):
        return self.step_fnc(batch=batch, batch_idx=batch_idx, stage="val")

    def on_validation_epoch_end(self):
        if not self.trainer.sanity_checking:
            # construct dict of logged metrics
            result_dict = {
                "epoch": float(self.current_epoch),
                "lr": self.trainer.optimizers[0].param_groups[0]["lr"],
            }
            result_dict.update(self._get_mean_loss_dict_for_type())
            self.log_dict(result_dict, sync_dist=True)

        self._reset_losses_dict()

    def test_step(self, batch, batch_idx):
        return self.step_fnc(batch=batch, batch_idx=batch_idx, stage="test")

    def on_test_epoch_end(self):
        # Log all test losses
        if not self.trainer.sanity_checking:
            result_dict = {}
            result_dict.update(self._get_mean_loss_dict_for_type())
            # Get only test entries
            result_dict = {k: v for k, v in result_dict.items() if k.startswith("test")}
            self.log_dict(result_dict, sync_dist=True)

            print(f"Test run finished!")

    def _reset_losses_dict(self):
        self.losses = {}
        for stage in ["train", "val", "test"]:
            self.losses[stage] = []

    def _get_mean_loss_dict_for_type(self):
        assert self.losses is not None
        mean_losses = {}
        for stage in ["train", "val", "test"]:
            if len(self.losses[stage]) > 0:
                mean_losses[stage] = torch.stack(self.losses[stage]).mean()
        return mean_losses

    def loss_non_nans(self, loss: Tensor, modality: str) -> Tensor:
        m = loss.isnan()
        if torch.any(m):
            print(f"Recovered NaNs in {modality}. Selecting NoN-Nans")
        return loss[~m]

    def step_fnc(self, batch, batch_idx, stage: str):
        batch_size = int(batch.batch.max()) + 1
        if stage == "train":
            t = torch.randint(
                low=1,
                high=self.hparams.timesteps + 1,
                size=(batch_size,),
                dtype=torch.long,
                device=self.device,
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
                weights = self.sde_atom_charge.exp_t_half_weighting(
                    t=t, device=self.device
                )
            elif self.hparams.loss_weighting == "uniform":
                weights = None
        else:
            t = torch.ones((batch_size,), device=self.device, dtype=torch.long)
            weights = None

        if self.hparams.context_mapping:
            context = prepare_context(
                self.hparams["properties_list"],
                self.prop_norm,
                batch,
                self.hparams.dataset,
            )
            batch.context = context

        out_dict = self(batch=batch, t=t)

        if self.hparams.regression_property == "sascore":
            labels = (
                torch.tensor([calculate_sascore(mol) for mol in batch.mol])
                .to(self.device)
                .float()
            )
        elif self.hparams.regression_property == "docking_score":
            labels = batch.docking_scores.float()

        if stage == "train" and weights is not None:
            loss = weights * self.loss(out_dict["property_pred"].squeeze(-1), labels)
        else:
            loss = self.loss(out_dict["property_pred"].squeeze(-1), labels)
        loss = torch.mean(loss, dim=0)

        self.losses[stage].append(loss.detach())

        self.log(
            f"{stage}/loss",
            loss,
            on_step=True,
            batch_size=batch_size,
            prog_bar=True,
            sync_dist=self.hparams.gpus > 1 and (stage == "val" or stage == "test"),
        )
        # if self.training:
        #     names = []
        #     loss.backward()
        #     for name, param in self.model.named_parameters():
        #         if param.grad is None:
        #             names.append(name)
        #     import pdb

        #     pdb.set_trace()
        return loss

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

        # out = self.model(
        #     x=atom_feats_in_perturbed, t=temb, pos=pos_perturbed, batch=batch_full
        # )
        out = self.model(
            x=atom_feats_in_perturbed,
            t=temb,
            pos=pos_perturbed,
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
        )

        return out

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.hparams["lr"],
            amsgrad=True,
            weight_decay=1.0e-12,
        )
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
