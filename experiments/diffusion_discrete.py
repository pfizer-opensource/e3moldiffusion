import logging
import os
from datetime import datetime
from torch_scatter import scatter_mean
from typing import Optional, List, Tuple

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from e3moldiffusion.coordsatomsbonds import DenoisingEdgeNetwork
from e3moldiffusion.molfeat import get_bond_feature_dims
from experiments.data.abstract_dataset import AbstractDatasetInfos
from experiments.diffusion.categorical import CategoricalDiffusionKernel
from experiments.diffusion.continuous import DiscreteDDPM
from experiments.losses import DiffusionLoss
from experiments.molecule_utils import Molecule
from experiments.sampling.analyze import analyze_stability_for_molecules
from experiments.utils import (
    coalesce_edges,
    get_list_of_edge_adjs,
    load_model,
    zero_mean,
)
from torch import Tensor
from torch_geometric.data import Batch
from torch_geometric.nn import radius_graph
from torch_geometric.utils import dense_to_sparse, sort_edge_index
from tqdm import tqdm

from e3moldiffusion.coordsatomsbonds import DenoisingEdgeNetwork
from e3moldiffusion.molfeat import get_bond_feature_dims
from experiments.diffusion.continuous import DiscreteDDPM
from experiments.diffusion.categorical import CategoricalDiffusionKernel
from experiments.data.distributions import prepare_context

from experiments.molecule_utils import Molecule
from experiments.utils import (
    coalesce_edges,
    get_list_of_edge_adjs,
    zero_mean,
    load_model,
    load_bond_model,
)
from experiments.sampling.analyze import analyze_stability_for_molecules
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

        if self.hparams.load_ckpt_from_pretrained is not None:
            print("Loading from pre-trained model checkpoint...")

            self.model = load_model(
                self.hparams.load_ckpt_from_pretrained, dataset_info
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

        if self.hparams.bond_model_guidance:
            print("Using bond model guidance...")
            self.bond_model = load_bond_model(
                self.hparams.ckpt_bond_model, dataset_info
            )
            for param in self.bond_model.parameters():
                param.requires_grad = False
            self.bond_model.eval()

    def training_step(self, batch, batch_idx):
        return self.step_fnc(batch=batch, batch_idx=batch_idx, stage="train")

    def validation_step(self, batch, batch_idx):
        return self.step_fnc(batch=batch, batch_idx=batch_idx, stage="val")

    def on_validation_epoch_end(self):
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
        charge_types_integer = charge_types_integer - self.dataset_info.charge_offset
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
                device=self.device,
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

    def _reverse_sample_categorical(
        self,
        cat_kernel: CategoricalDiffusionKernel,
        xt: Tensor,
        x0: Tensor,
        t: Tensor,
        num_classes: int,
    ):
        reverse = cat_kernel.reverse_posterior_for_every_x0(xt=xt, t=t)
        # Eq. 4 in Austin et al. (2023) "Structured Denoising Diffusion Models in Discrete State-Spaces"
        # (N, a_0, a_t-1)
        unweighted_probs = (reverse * x0.unsqueeze(-1)).sum(1)
        unweighted_probs[unweighted_probs.sum(dim=-1) == 0] = 1e-5
        # (N, a_t-1)
        probs = unweighted_probs / unweighted_probs.sum(-1, keepdims=True)
        x_tm1 = F.one_hot(
            probs.multinomial(
                1,
            ).squeeze(),
            num_classes=num_classes,
        ).float()
        return x_tm1

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

        # edge types for FC graph
        edge_index_global = (
            torch.eq(batch.unsqueeze(0), batch.unsqueeze(-1)).int().fill_diagonal_(0)
        )
        edge_index_global, _ = dense_to_sparse(edge_index_global)
        edge_index_global = sort_edge_index(edge_index_global, sort_by_row=False)
        j, i = edge_index_global
        mask = j < i
        mask_i = i[mask]
        mask_j = j[mask]
        nE = len(mask_i)
        edge_attr_triu = torch.multinomial(
            self.bonds_prior, num_samples=nE, replacement=True
        )

        j = torch.concat([mask_j, mask_i])
        i = torch.concat([mask_i, mask_j])
        edge_index_global = torch.stack([j, i], dim=0)
        edge_attr_global = torch.concat([edge_attr_triu, edge_attr_triu], dim=0)
        edge_index_global, edge_attr_global = sort_edge_index(
            edge_index=edge_index_global, edge_attr=edge_attr_global, sort_by_row=False
        )
        j, i = edge_index_global
        mask = j < i
        mask_i = i[mask]
        mask_j = j[mask]

        # some assert

        edge_attr_global_dense = torch.zeros(
            size=(n, n), device=device, dtype=torch.long
        )
        edge_attr_global_dense[
            edge_index_global[0], edge_index_global[1]
        ] = edge_attr_global
        assert (edge_attr_global_dense - edge_attr_global_dense.T).sum().float() == 0.0

        edge_attr_global = F.one_hot(edge_attr_global, self.num_bond_classes).float()

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
                pos=pos,
                edge_index_local=edge_index_local,
                edge_index_global=edge_index_global,
                edge_attr_global=edge_attr_global,
                batch=batch,
                batch_edge_global=batch_edge_global,
                context=context,
            )

            coords_pred = out["coords_pred"].squeeze()
            if self.hparams.noise_scheduler == "adaptive":
                # # Sample the positions
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

                mu = z_t_prefactor[batch] * pos + x_prefactor[batch] * coords_pred

                noise = torch.randn_like(pos)
                noise = zero_mean(noise, batch=batch, dim_size=bs, dim=0)

                pos = mu + noise_prefactor[batch] * noise

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

                # positions/coords
                mean = (
                    sqrt_alphas[batch] * one_m_alphas_cumprod_prev[batch] * pos
                    + sqrt_alphas_cumprod_prev[batch]
                    * one_m_alphas[batch]
                    * coords_pred
                )
                mean = (1.0 / sigmas2t[batch]) * mean
                std = rev_sigma[batch]
                noise = torch.randn_like(mean)
                noise = zero_mean(noise, batch=batch, dim_size=bs, dim=0)
                pos = mean + std * noise

            atoms_pred, charges_pred = out["atoms_pred"].split(
                [self.num_atom_types, self.num_charge_classes], dim=-1
            )
            atoms_pred = atoms_pred.softmax(dim=-1)
            # N x a_0
            edges_pred = out["bonds_pred"].softmax(dim=-1)
            # E x b_0
            charges_pred = charges_pred.softmax(dim=-1)

            # atoms
            atom_types = self._reverse_sample_categorical(
                cat_kernel=self.cat_atoms,
                xt=atom_types,
                x0=atoms_pred,
                t=t[batch],
                num_classes=self.num_atom_types,
            )

            # charges
            charge_types = self._reverse_sample_categorical(
                cat_kernel=self.cat_charges,
                xt=charge_types,
                x0=charges_pred,
                t=t[batch],
                num_classes=self.num_charge_classes,
            )

            # edges
            edges_pred_triu = edges_pred[mask]
            # (E, b_0)
            edges_triu = edge_attr_global[mask]

            edges_triu = self._reverse_sample_categorical(
                cat_kernel=self.cat_bonds,
                xt=edges_triu,
                x0=edges_pred_triu,
                t=t[batch[mask_i]],
                num_classes=self.num_bond_classes,
            )

            j, i = edge_index_global
            mask = j < i
            mask_i = i[mask]
            mask_j = j[mask]
            j = torch.concat([mask_j, mask_i])
            i = torch.concat([mask_i, mask_j])
            edge_index_global = torch.stack([j, i], dim=0)
            edge_attr_global = torch.concat([edges_triu, edges_triu], dim=0)
            edge_index_global, edge_attr_global = sort_edge_index(
                edge_index=edge_index_global,
                edge_attr=edge_attr_global,
                sort_by_row=False,
            )

            if not self.hparams.fully_connected:
                edge_index_local = radius_graph(
                    x=pos.detach(),
                    r=self.hparams.cutoff_local,
                    batch=batch,
                    max_num_neighbors=self.hparams.max_num_neighbors,
                )

            # atom_integer = torch.argmax(atom_types, dim=-1)
            # bond_integer = torch.argmax(edge_attr_global, dim=-1)

            if self.hparams.bond_model_guidance:
                guidance_type = "logsum"
                guidance_scale = 1.0e-4
                with torch.enable_grad():
                    node_feats_in = node_feats_in.detach()
                    pos = pos.detach().requires_grad_(True)
                    bond_prediction = self.bond_model(
                        x=node_feats_in,
                        t=temb,
                        pos=pos,
                        edge_index_local=edge_index_local,
                        edge_index_global=edge_index_global,
                        edge_attr_global=edge_attr_global,
                        batch=batch,
                        batch_edge_global=batch_edge_global,
                    )
                    if guidance_type == "ensemble":
                        # TO-DO
                        raise NotImplementedError
                    elif guidance_type == "logsum":
                        uncertainty = torch.sigmoid(
                            -torch.logsumexp(bond_prediction, dim=-1)
                        )
                        uncertainty = (
                            0.5
                            * scatter_mean(
                                uncertainty,
                                index=edge_index_global[1],
                                dim=0,
                                dim_size=pos.size(0),
                            ).log()
                        )
                        uncertainty = scatter_mean(
                            uncertainty, index=batch, dim=0, dim_size=bs
                        )
                        grad_outputs: List[Optional[torch.Tensor]] = [
                            torch.ones_like(uncertainty)
                        ]
                        dist_shift = -torch.autograd.grad(
                            [uncertainty],
                            [pos],
                            grad_outputs=grad_outputs,
                            create_graph=False,
                            retain_graph=False,
                        )[0]

                pos = pos + guidance_scale * dist_shift

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
