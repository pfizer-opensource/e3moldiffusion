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
import numpy as np


def get_sigmas(sigma_begin, sigma_end, num_noise_scales: int = 10):
    sigmas = (
        torch.tensor(
            np.exp(
                np.linspace(
                    np.log(sigma_begin),
                    np.log(sigma_end),
                    num_noise_scales,
                )
            )
        ).float()
    )
    return sigmas


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
        
        self.hparams.num_atom_types = dataset_info.input_dims.X
        self.num_charge_classes = dataset_info.input_dims.C
        self.num_atom_types = self.hparams.num_atom_types
        self.num_atom_features = self.num_atom_types + self.num_charge_classes

        self.sigma_min = 0.01
        self.sigma_max = 0.10
        sigmas = get_sigmas(sigma_begin=self.sigma_min, sigma_end=self.sigma_max, num_noise_scales=10)
        self.register_buffer("sigmas", sigmas)
        
        if hparams.get("no_h"):
            print("Training without hydrogen")

        self.model = EQGATForceNetwork(
            hn_dim=(hparams["sdim"], hparams["vdim"]),
            num_layers=hparams["num_layers"],
            num_rbfs=hparams["rbf_dim"],
            edge_dim=None,
            use_cross_product=hparams["use_cross_product"],
            num_atom_features=self.num_atom_features,
            cutoff_local=hparams["cutoff_local"],
            vector_aggr="add",
        )
     
    def loss_non_nans(self, loss: Tensor, modality: str) -> Tensor:
        m = loss.isnan()
        if torch.any(m):
            print(f"Recovered NaNs in {modality}. Selecting NoN-Nans")
        return loss[~m]  

    def training_step(self, batch, batch_idx):
        return self.step_fnc(batch=batch, batch_idx=batch_idx, stage="train")

    def validation_step(self, batch, batch_idx):
        return self.step_fnc(batch=batch, batch_idx=batch_idx, stage="val")
   
    def step_fnc(self, batch, batch_idx, stage: str, anneal_power=2.):
    
        out_dict, used_sigmas, noise = self(batch=batch, fitting=True)        
        scores = out_dict["pseudo_forces_pred"] / used_sigmas
        target = -1.0 / (used_sigmas**2) * noise
        target = target.view(target.shape[0], -1)
        scores = scores.view(scores.shape[0], -1)
        loss = 1 / 2. * ((scores - target) ** 2).sum(dim=-1) * used_sigmas.squeeze() ** anneal_power
        # loss = scatter_mean(loss, index=batch.batch, dim=0, dim_size=len(batch.batch.unique()))
        loss = self.loss_non_nans(loss, 'pseudo-forces')
        loss = torch.mean(loss, dim=0)
        
        self.log(
            f"{stage}/loss",
            loss,
            on_step=True,
            batch_size=len(batch.batch.unique()),
            prog_bar=True,
            sync_dist=self.hparams.gpus > 1 and stage == "val",
        )

        return loss

    def forward(self, batch: Batch, fitting=True):
        atom_types: Tensor = batch.x
        pos: Tensor = batch.pos
        charges: Tensor = batch.charges
        data_batch: Tensor = batch.batch
       
        bs = int(data_batch.max()) + 1
        
        edge_index_local = radius_graph(
            x=pos,
            r=self.hparams.cutoff_local,
            batch=data_batch,
            max_num_neighbors=128,
            flow="source_to_target",
        )
        
        edge_attr_local = None
        
        atom_types = F.one_hot(atom_types.squeeze().long(), num_classes=self.num_atom_types_geom).float()
        charges = self.dataset_info.one_hot_charges(charges).float()
        
        atom_feats_in = torch.cat(
            [atom_types, charges], dim=-1
        )
        
        labels = torch.randint(0, len(self.sigmas), (bs,), device=batch.x.device)
        used_sigmas = self.sigmas[labels][batch.batch].unsqueeze(1)        
        pos_centered = zero_mean(batch.pos, batch.batch, dim=0, dim_size=bs)      
        noise = torch.randn_like(pos_centered)
        noise = zero_mean(noise, batch.batch, dim=0, dim_size=bs)
        noise = used_sigmas * noise
        pos_perturbed = pos_centered + noise
        out = self.model(
            x=atom_feats_in, pos=pos_perturbed, batch=data_batch,
            edge_index=edge_index_local, edge_attr=edge_attr_local
        )
        
        return out, used_sigmas, noise

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.hparams["lr"],
            amsgrad=True,
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