import logging
import os
from datetime import datetime
from typing import List, Tuple
from tqdm import tqdm
import pandas as pd

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Batch
from torch_geometric.nn import radius_graph
from torch_geometric.utils import dense_to_sparse, sort_edge_index

from e3moldiffusion.coordsatomsbonds import DenoisingEdgeNetwork
from e3moldiffusion.molfeat import get_bond_feature_dims
from experiments.diffusion.continuous import DiscreteDDPM
from experiments.data.abstract_dataset import AbstractDatasetInfos
from experiments.data.distributions import prepare_context

from experiments.molecule_utils import Molecule
from experiments.utils import (
    coalesce_edges,
    get_empirical_num_nodes,
    get_list_of_edge_adjs,
    zero_mean,
)
from experiments.sampling.analyze import analyze_stability_for_molecules

from experiments.losses import DiffusionLoss

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
    ):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.i = 0

        self.dataset_info = dataset_info

        self.prop_norm = prop_norm
        self.prop_dist = prop_dist

        atom_types_distribution = dataset_info.atom_types.float()
        bond_types_distribution = dataset_info.edge_types.float()
        charge_types_distribution = dataset_info.charges_marginals.float()

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
        )

        self.sde = DiscreteDDPM(
            beta_min=hparams["beta_min"],
            beta_max=hparams["beta_max"],
            N=hparams["timesteps"],
            scaled_reverse_posterior_sigma=True,
            schedule="cosine",
            enforce_zero_terminal_snr=False,
        )

        self.diffusion_loss = DiffusionLoss(
            modalities=["coords", "atoms", "charges", "bonds"]
        )

    def training_step(self, batch, batch_idx):
        return self.step_fnc(batch=batch, batch_idx=batch_idx, stage="train")

    def validation_step(self, batch, batch_idx):
        return self.step_fnc(batch=batch, batch_idx=batch_idx, stage="val")

    def on_validation_epoch_end(self):
        if (self.current_epoch + 1) % self.hparams.test_interval == 0:
            print(f"Running evaluation in epoch {self.current_epoch + 1}")
            final_res = self.run_evaluation(
                step=self.i,
                dataset_info=self.dataset_info,
                ngraphs=1000,
                bs=self.hparams.inference_batch_size,
                verbose=True,
                inner_verbose=False,
            )
            self.i += 1
            self.log(name="val/validity", value=final_res.validity[0], on_epoch=True)
            self.log(
                name="val/uniqueness", value=final_res.uniqueness[0], on_epoch=True
            )
            self.log(name="val/novelty", value=final_res.novelty[0], on_epoch=True)
            self.log(
                name="val/mol_stable", value=final_res.mol_stable[0], on_epoch=True
            )
            self.log(
                name="val/atm_stable", value=final_res.atm_stable[0], on_epoch=True
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
        if self.hparams.context_mapping:
            context = prepare_context(
                self.hparams["properties_list"],
                self.prop_norm,
                batch,
                self.hparams.dataset,
            )
            batch.context = context

        if self.hparams.use_loss_weighting:
            weights = torch.clip(torch.exp(-t / 200), min=0.1).to(self.device)
        else:
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
        context = batch.context if self.hparams.context_mapping else None
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

        # create block diagonal matrix
        dense_edge = torch.zeros(n, n, device=pos.device, dtype=torch.long)
        # populate entries with integer features
        dense_edge[edge_index_global[0, :], edge_index_global[1, :]] = edge_attr_global
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

        signal = self.sde.sqrt_alphas_cumprod[t]
        std = self.sde.sqrt_1m_alphas_cumprod[t]

        signal_b = signal[data_batch].unsqueeze(-1).unsqueeze(-1)
        std_b = std[data_batch].unsqueeze(-1).unsqueeze(-1)
        dense_edge_ohe_perturbed = dense_edge_ohe * signal_b + noise_edges * std_b

        # retrieve as edge-attributes in PyG Format
        edge_attr_global_perturbed = dense_edge_ohe_perturbed[
            edge_index_global[0, :], edge_index_global[1, :], :
        ]
        # edge_attr_global_noise = noise_edges[edge_index_global[0, :], edge_index_global[1, :], :]

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

        # Atom-types
        if self.hparams.no_h:
            node_feat -= 1

        atom_types = F.one_hot(atom_types, num_classes=self.num_atom_types).float()
        # sample noise for OHEs in {0, 1}^NUM_CLASSES
        noise_atom_types = torch.randn_like(atom_types)
        mean_ohes, std_ohes = self.sde.marginal_prob(x=atom_types, t=t[data_batch])
        # perturb OHEs
        atom_types_perturbed = mean_ohes + std_ohes * noise_atom_types

        # Charges
        charges = self.dataset_statistics.one_hot_charges(charges).float()
        # sample noise for OHEs in {0, 1}^NUM_CLASSES
        noise_charges = torch.randn_like(charges)
        mean_ohes, std_ohes = self.sde.marginal_prob(x=charges, t=t[data_batch])
        # perturb OHEs
        charges_perturbed = mean_ohes + std_ohes * noise_charges

        if not self.hparams.fully_connected:
            edge_index_local = radius_graph(
                x=pos_perturbed,
                r=self.hparams.cutoff_local,
                batch=data_batch,
                flow="source_to_target",
                max_num_neighbors=self.hparams.max_num_neighbors,
            )
        else:
            edge_index_local = None

        batch_edge_global = data_batch[edge_index_global[0]]

        atom_feats_in_perturbed = torch.cat(
            [atom_types_perturbed, charges_perturbed], dim=-1
        )

        out = self.model(
            x=atom_feats_in_perturbed,
            t=temb,
            pos=pos_perturbed,
            edge_index_local=edge_index_local,
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

        out["coords_true"] = pos_centered
        out["atoms_true"] = atom_types.argmax(dim=-1)
        out["bonds_true"] = edge_attr_global
        out["charges_true"] = charges.argmax(dim=-1)

        out["bond_aggregation_index"] = edge_index_global[1]

        return out

    @torch.no_grad()
    def generate_graphs(
        self,
        num_graphs: int,
        empirical_distribution_num_nodes: torch.Tensor,
        device: torch.device,
        verbose=False,
        save_traj=False,
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
            num_graphs=num_graphs,
            device=device,
            empirical_distribution_num_nodes=empirical_distribution_num_nodes,
            verbose=verbose,
            save_traj=save_traj,
        )

        pos_splits = pos.detach().split(batch_num_nodes.cpu().tolist(), dim=0)

        charge_types_integer = torch.argmax(charge_types, dim=-1)
        # offset back
        charge_types_integer = (
            charge_types_integer - self.dataset_statistics.charge_offset
        )
        charge_types_integer_split = charge_types_integer.detach().split(
            batch_num_nodes.cpu().tolist(), dim=0
        )
        atom_types_integer = torch.argmax(atom_types, dim=-1)
        if self.hparams.no_h:
            raise NotImplementedError  # remove in future or implement
            atom_types_integer += 1

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
        return_smiles: bool = False,
        verbose: bool = False,
        inner_verbose=False,
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
            (
                pos_splits,
                atom_types_integer_split,
                charge_types_integer_split,
                edge_types,
                edge_index_global,
                batch_num_nodes,
                _,
            ) = self.generate_graphs(
                num_graphs=num_graphs,
                verbose=inner_verbose,
                device=self.empirical_num_nodes.device,
                empirical_distribution_num_nodes=self.empirical_num_nodes,
                save_traj=False,
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
        ) = analyze_stability_for_molecules(
            molecule_list=molecule_list,
            dataset_info=dataset_info,
            smiles_train=self.smiles_list,
            local_rank=self.local_rank,
            return_smiles=return_smiles,
            device=self.device,
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
            with open(save_dir, "a") as f:
                total_res.to_csv(f, header=True)
        except Exception as e:
            print(e)
            pass

        if return_smiles:
            return total_res, all_generated_smiles
        else:
            return total_res

    def reverse_sampling(
        self,
        num_graphs: int,
        empirical_distribution_num_nodes: Tensor,
        device: torch.device,
        verbose: bool = False,
        save_traj: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, List]:
        batch_num_nodes = torch.multinomial(
            input=empirical_distribution_num_nodes,
            num_samples=num_graphs,
            replacement=True,
        ).to(device)
        batch_num_nodes = batch_num_nodes.clamp(min=1)
        batch = torch.arange(num_graphs, device=device).repeat_interleave(
            batch_num_nodes, dim=0
        )
        bs = int(batch.max()) + 1

        # sample context condition
        context = None
        if self.prop_dist is not None:
            context = self.prop_dist.sample_batch(batch_num_nodes).to(self.device)[
                batch
            ]

        # initialiaze the 0-mean point cloud from N(0, I)
        pos = torch.randn(len(batch), 3, device=device, dtype=torch.get_default_dtype())
        pos = zero_mean(pos, batch=batch, dim_size=bs, dim=0)

        # initialize the atom-types
        atom_types = torch.randn(pos.size(0), self.num_atom_types, device=device)

        # initialize the charges
        charge_types = torch.randn(pos.size(0), self.num_charge_classes, device=device)

        # edge_index_local = radius_graph(x=pos,
        #                                r=self.hparams.cutoff_local,
        #                                batch=batch,
        #                                max_num_neighbors=self.hparams.max_num_neighbors)
        edge_index_local = None

        edge_index_global = (
            torch.eq(batch.unsqueeze(0), batch.unsqueeze(-1)).int().fill_diagonal_(0)
        )
        # sample symmetric edge-attributes
        edge_attrs = torch.randn(
            (
                edge_index_global.size(0),
                edge_index_global.size(1),
                self.num_bond_classes,
            ),
            device=device,
            dtype=torch.get_default_dtype(),
        )
        # symmetrize
        edge_attrs = 0.5 * (edge_attrs + edge_attrs.permute(1, 0, 2))
        assert torch.norm(edge_attrs - edge_attrs.permute(1, 0, 2)).item() == 0.0
        # get COO format (2, E)
        edge_index_global, _ = dense_to_sparse(edge_index_global)
        edge_index_global = sort_edge_index(edge_index_global, sort_by_row=False)
        # select in PyG formt (E, self.hparams.num_bond_types)
        edge_attr_global = edge_attrs[
            edge_index_global[0, :], edge_index_global[1, :], :
        ]
        batch_edge_global = batch[edge_index_global[0]]

        pos_traj = []
        atom_type_traj = []
        charge_type_traj = []
        edge_type_traj = []

        chain = range(self.hparams.timesteps)

        iterator = (
            tqdm(reversed(chain), total=len(chain)) if verbose else reversed(chain)
        )
        for timestep in iterator:
            t = torch.full(
                size=(bs,), fill_value=timestep, dtype=torch.long, device=pos.device
            )
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

            rev_sigma = self.sde.reverse_posterior_sigma[t].unsqueeze(-1)
            sigmast = self.sde.sqrt_1m_alphas_cumprod[t].unsqueeze(-1)
            sigmas2t = sigmast.pow(2)

            sqrt_alphas = self.sde.sqrt_alphas[t].unsqueeze(-1)
            sqrt_1m_alphas_cumprod_prev = torch.sqrt(
                1.0 - self.sde.alphas_cumprod_prev[t]
            ).unsqueeze(-1)
            one_m_alphas_cumprod_prev = sqrt_1m_alphas_cumprod_prev.pow(2)
            sqrt_alphas_cumprod_prev = torch.sqrt(
                self.sde.alphas_cumprod_prev[t].unsqueeze(-1)
            )
            one_m_alphas = self.sde.discrete_betas[t].unsqueeze(-1)

            coords_pred = out["coords_pred"].squeeze()
            atoms_pred, charges_pred = out["atoms_pred"].split(
                [self.num_atom_types, self.num_charge_classes], dim=-1
            )
            atoms_pred = atoms_pred.softmax(dim=-1)
            # N x a_0
            edges_pred = out["bonds_pred"].softmax(dim=-1)
            # E x b_0
            charges_pred = charges_pred.softmax(dim=-1)

            # update fnc

            # positions/coords
            mean = (
                sqrt_alphas[batch] * one_m_alphas_cumprod_prev[batch] * pos
                + sqrt_alphas_cumprod_prev[batch] * one_m_alphas[batch] * coords_pred
            )
            mean = (1.0 / sigmas2t[batch]) * mean
            std = rev_sigma[batch]
            noise = torch.randn_like(mean)
            noise = zero_mean(noise, batch=batch, dim_size=bs, dim=0)
            pos = mean + std * noise

            # atoms
            mean = (
                sqrt_alphas[batch] * one_m_alphas_cumprod_prev[batch] * atom_types
                + sqrt_alphas_cumprod_prev[batch] * one_m_alphas[batch] * atoms_pred
            )
            mean = (1.0 / sigmas2t[batch]) * mean
            std = rev_sigma[batch]
            noise = torch.randn_like(mean)
            atom_types = mean + std * noise

            # charges
            mean = (
                sqrt_alphas[batch] * one_m_alphas_cumprod_prev[batch] * charge_types
                + sqrt_alphas_cumprod_prev[batch] * one_m_alphas[batch] * charges_pred
            )
            mean = (1.0 / sigmas2t[batch]) * mean
            std = rev_sigma[batch]
            noise = torch.randn_like(mean)
            charge_types = mean + std * noise

            # edges
            mean = (
                sqrt_alphas[batch_edge_global]
                * one_m_alphas_cumprod_prev[batch_edge_global]
                * edge_attr_global
                + sqrt_alphas_cumprod_prev[batch_edge_global]
                * one_m_alphas[batch_edge_global]
                * edges_pred
            )
            mean = (1.0 / sigmas2t[batch_edge_global]) * mean
            std = rev_sigma[batch_edge_global]
            noise_edges = torch.randn_like(edge_attrs)
            noise_edges = 0.5 * (noise_edges + noise_edges.permute(1, 0, 2))
            noise_edges = noise_edges[
                edge_index_global[0, :], edge_index_global[1, :], :
            ]
            edge_attr_global = mean + std * noise_edges

            if not self.hparams.fully_connected:
                edge_index_local = radius_graph(
                    x=pos.detach(),
                    r=self.hparams.cutoff_local,
                    batch=batch,
                    max_num_neighbors=self.hparams.max_num_neighbors,
                )

                # include local (noisy) edge-attributes based on radius graph indices
                edge_attrs = torch.zeros_like(edge_attrs)
                edge_attrs[
                    edge_index_global[0], edge_index_global[1], :
                ] = edge_attr_global

            # atom_integer = torch.argmax(atom_types, dim=-1)
            # bond_integer = torch.argmax(edge_attr_global, dim=-1)

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
        # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        #    optimizer=optimizer,
        #    gamma=self.hparams.gamma,
        #    last_epoch=-1
        # )
        # scheduler = {
        #    "scheduler": lr_scheduler,
        #    "interval": "epoch",
        #    "frequency": self.hparams["lr_frequency"],
        #    "monitor": "val/loss",
        #    "strict": False,
        # }
        return [optimizer]  # , [scheduler]
