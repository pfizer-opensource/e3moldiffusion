import logging
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Batch
from torch_geometric.nn import radius_graph

from experiments.diffusion.continuous import DiscreteDDPM
from experiments.diffusion.categorical import CategoricalDiffusionKernel
from e3moldiffusion.coordsatomsbonds import EQGATForceNetwork
from experiments.data.abstract_dataset import AbstractDatasetInfos
from experiments.utils import coalesce_edges, zero_mean
from torch_scatter import scatter_add, scatter_mean

class Trainer(pl.LightningModule):
    def __init__(
        self,
        hparams: dict,
        dataset_info: AbstractDatasetInfos,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.dataset_info = dataset_info

        self.num_atom_types_geom = 16
        self.num_bond_classes = 5
        atom_types_distribution = dataset_info.atom_types.float()
        bond_types_distribution = dataset_info.edge_types.float()
        charge_types_distribution = dataset_info.charges_marginals.float()

        self.register_buffer("atoms_prior", atom_types_distribution.clone())
        self.register_buffer("bonds_prior", bond_types_distribution.clone())
        self.register_buffer("charges_prior", charge_types_distribution.clone())

        self.hparams.num_atom_types = dataset_info.input_dims.X
        self.num_charge_classes = dataset_info.input_dims.C
        self.num_atom_types = self.hparams.num_atom_types
        self.num_atom_features = self.num_atom_types + self.num_charge_classes

        if hparams.get("no_h"):
            print("Training without hydrogen")

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

        self.cat_atoms = CategoricalDiffusionKernel(
            terminal_distribution=atom_types_distribution,
            alphas=self.sde_atom_charge.alphas.clone(),
        )
        self.cat_charges = CategoricalDiffusionKernel(
            terminal_distribution=charge_types_distribution,
            alphas=self.sde_atom_charge.alphas.clone(),
        )

        self.model = EQGATForceNetwork(
            hn_dim=(hparams["sdim"], hparams["vdim"]),
            num_layers=hparams["num_layers"],
            num_rbfs=hparams["rbf_dim"],
            edge_dim=hparams["rbf_dim"],
            use_cross_product=hparams["use_cross_product"],
            num_atom_features=self.num_atom_features,
            cutoff_local=hparams["cutoff_local"],
            vector_aggr=hparams["vector_aggr"],
        )

        self.force_loss = torch.nn.MSELoss(reduce=False, reduction="none")
       
    def training_step(self, batch, batch_idx):
        return self.step_fnc(batch=batch, batch_idx=batch_idx, stage="train")

    def validation_step(self, batch, batch_idx):
        return self.step_fnc(batch=batch, batch_idx=batch_idx, stage="val")

    def loss_non_nans(self, loss: Tensor, modality: str) -> Tensor:
        m = loss.isnan()
        if torch.any(m):
            print(f"Recovered NaNs in {modality}. Selecting NoN-Nans")
        return loss[~m]
    
    def step_fnc(self, batch, batch_idx, stage: str):
        
        is_train = stage == "train"
        out_dict, t, batch_size = self(batch=batch, train=is_train)

        if self.hparams.loss_weighting == "snr_s_t":
            weights = self.sde_atom_charge.snr_s_t_weighting(
                s=t - 1, t=t, device=self.device, clamp_min=0.05, clamp_max=5.0
            )
        elif self.hparams.loss_weighting == "snr_t":
            weights = self.sde_atom_charge.snr_t_weighting(
                t=t,
                device=self.device,
                clamp_min=self.hparams.snr_clamp_min,
                clamp_max=self.hparams.snr_clamp_max,
            )
        elif self.hparams.loss_weighting == "exp_t":
            weights = self.sde_atom_charge.exp_t_weighting(t=t, device=self.device)
        elif self.hparams.loss_weighting == "exp_t_half":
            weights = self.sde_atom_charge.exp_t_half_weighting(t=t, device=self.device)
        elif self.hparams.loss_weighting == "uniform":
            weights = torch.ones((batch_size,), device=self.device)

        #import pdb
        #pdb.set_trace()
        loss = self.force_loss(out_dict["pseudo_forces_pred"], out_dict["pseudo_forces_true"])
        if loss.ndim == 2:
            loss = loss.mean(dim=1)
        loss = scatter_mean(loss, index=batch.batch, dim=0, dim_size=batch_size)
        loss = weights * loss
        loss = self.loss_non_nans(loss=loss, modality="forces")
        loss = torch.mean(loss, dim=0)

        self.log(
            f"{stage}/loss",
            loss,
            on_step=True,
            batch_size=batch_size,
            prog_bar=True,
            sync_dist=self.hparams.gpus > 1 and stage == "val",
        )

        return loss

    def forward(self, batch: Batch, train: bool = True):
        atom_types: Tensor = batch.x
        pos: Tensor = batch.pos
        charges: Tensor = batch.charges
        data_batch: Tensor = batch.batch
        n = batch.num_nodes
        bond_edge_index = batch.edge_index
        bond_edge_attr = batch.edge_attr
        
        bs = int(data_batch.max()) + 1
        
        t = torch.randint(
            low=1,
            high=self.hparams.timesteps + 1,
            size=(bs,),
            dtype=torch.long,
            device=batch.x.device,
        )
        
        if not train:
            t = torch.zeros_like(t)

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
        
        pos_centered = zero_mean(pos, data_batch, dim=0, dim_size=bs)

        # SAMPLING
        noise_coords_true, pos_perturbed = self.sde_pos.sample_pos(
            t, pos_centered, data_batch
        )
        atom_types, _ = self.cat_atoms.sample_categorical(
            t, atom_types, data_batch, self.dataset_info, type="atoms"
        )
        charges, _ = self.cat_charges.sample_categorical(
            t, charges, data_batch, self.dataset_info, type="charges"
        )
        
        atom_feats_in = torch.cat(
            [atom_types, charges], dim=-1
        )

        # TIME EMBEDDING
        temb = t.float() / self.hparams.timesteps
        temb = temb.clamp(min=self.hparams.eps_min)
        temb = temb.unsqueeze(dim=1)

        out = self.model(
            x=atom_feats_in, t=temb, pos=pos_perturbed, batch=data_batch,
            edge_index=edge_index_local, edge_attr=edge_attr_local
        )
        out["pseudo_forces_true"] = noise_coords_true

        return out, t, bs

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.hparams["lr"],
            amsgrad=False,
            weight_decay=1e-6,
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