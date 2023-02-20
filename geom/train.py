import logging
import os
from argparse import ArgumentParser
from typing import List, Optional, Tuple

import pytorch_lightning as pl
import torch
from geom.data import MolFeaturization
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger
from e3moldiffusion.molfeat import get_bond_feature_dims
from e3moldiffusion.sde import VPSDE, VPAncestralSamplingPredictor, get_timestep_embedding, ChebyshevExpansion, DiscreteDDPM
from e3moldiffusion.gnn import EQGATEncoder
import numpy as np

from torch import Tensor
from torch_geometric.data import Batch
from torch_geometric.typing import OptTensor
from torch_cluster import radius_graph
from torch_sparse import coalesce
from torch_scatter import scatter_mean
from tqdm import tqdm

logging.getLogger("lightning").setLevel(logging.WARNING)

BOND_FEATURE_DIMS = get_bond_feature_dims()[0]

default_hparams = {
    "sdim": 64,
    "vdim": 16,
    "tdim": 64,
    "edim": 16,
    "num_layers": 5,
    "num_diffusion_timesteps": 1000,
    "beta_min": 0.1,
    "beta_max": 20.0,
    "energy_preserving": False,
    "batch_size": 256,
    "dataset": "drugs",
    "save_dir": "diffusion",
    "omit_norm": False,
}

class Trainer(pl.LightningModule):
    def __init__(self, hparams=default_hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        self._hparams = hparams
        
        self.model = EQGATEncoder(
            hn_dim=(hparams["sdim"], hparams["vdim"]),
            t_dim=hparams["tdim"],
            edge_dim=hparams["edim"],
            num_layers=hparams["num_layers"],
            energy_preserving=hparams["energy_preserving"],
            use_norm=not hparams["omit_norm"],
            use_cross_product=not hparams["omit_cross_product"],
            use_mlp_update=False,
            use_all_atom_features=hparams["use_all_atom_features"]
        )

        timesteps = torch.arange(hparams["num_diffusion_timesteps"], dtype=torch.long)
        timesteps_embedder = get_timestep_embedding(
            timesteps=timesteps, embedding_dim=hparams["tdim"]
        ).to(torch.float32)

        self.register_buffer("timesteps", tensor=timesteps)
        self.register_buffer("timesteps_embedder", tensor=timesteps_embedder)
        
        if hparams["continuous"]:
            self.sde = VPSDE(beta_min=hparams["beta_min"], beta_max=hparams["beta_max"],
                             N=hparams["num_diffusion_timesteps"],
                             scaled_reverse_posterior_sigma=True)
        else:
            self.sde = DiscreteDDPM(beta_min=hparams["beta_min"], beta_max=hparams["beta_max"],
                                    N=hparams["num_diffusion_timesteps"],
                                    scaled_reverse_posterior_sigma=True)
            
        self.sampler = VPAncestralSamplingPredictor(sde=self.sde)
        

    def reverse_sampling(
        self,
        x: Tensor,
        edge_index: OptTensor = None,
        edge_attr: OptTensor = None,
        batch: OptTensor = None,
        num_diffusion_timesteps: Optional[int] = None,
        save_traj: bool = False,
        bond_edge_index: OptTensor = None,
        bond_edge_attr: OptTensor = None
    ) -> Tuple[Tensor, List]:
                
        if num_diffusion_timesteps is None:
            num_diffusion_timesteps = self._hparams["num_diffusion_timesteps"]

        if batch is None:
            batch = torch.zeros(len(x), device=x.device, dtype=torch.long)

        if self._hparams["use_bond_features"]:
            assert bond_edge_index is not None
            assert bond_edge_attr is not None
            
        bs = int(batch.max()) + 1
                
        # initialiaze the 0-mean point cloud from N(0, I)
        pos = torch.randn(x.size(0), 3,
                          device=x.device,
                          dtype=self.model.atom_encoder.atom_embedding_list[0].weight.dtype)
        pos = pos - scatter_mean(pos, dim=0, index=batch, dim_size=bs)[batch]
        
        if self._hparams["fully_connected"]:
            if edge_index is None:
                # fully-connected graphs
                batch_num_nodes = torch.bincount(batch)
                ptr = torch.cumsum(batch_num_nodes, dim=0)
                ptr = torch.concat([torch.zeros(1, device=ptr.device, dtype=ptr.dtype), ptr[:-1]], dim=0)       
                edge_index_list = []
                for offset, n in zip(ptr.cpu().tolist(), batch_num_nodes.cpu().tolist()):
                    row = torch.arange(n, dtype=torch.long)
                    col = torch.arange(n, dtype=torch.long)
                    row = row.view(-1, 1).repeat(1, n).view(-1)
                    col = col.repeat(n)
                    edge_index = torch.stack([row, col], dim=0)
                    mask = edge_index[0] != edge_index[1]
                    edge_index = edge_index[:, mask]
                    edge_index += offset
                    edge_index_list.append(edge_index)
                edge_index = torch.concat(edge_index_list, dim=-1).to(x.device)
        else:
            edge_index = radius_graph(x=pos, r=self._hparams["cutoff"], batch=batch, max_num_neighbors=64)
        
        if self._hparams["use_bond_features"]:
            # possibly combine the bond-edge-index with radius graph or fully-connected graph
            # Note: This scenario is useful when learning the 3D coordinates only. 
            # From an optimization perspective, atoms that are connected by topology should have certain distance values. 
            # Since the atom types are fixed here, we know which molecule we want to generate a 3D configuration from, so the edge-index will help as inductive bias
            edge_attr = torch.full(size=(edge_index.size(-1), ), fill_value=BOND_FEATURE_DIMS + 1, device=edge_index.device, dtype=torch.long)
            # combine
            edge_index = torch.cat([edge_index, bond_edge_index], dim=-1)
            edge_attr =  torch.cat([edge_attr, bond_edge_attr], dim=0)
            # coalesce, i.e. reduce and remove duplicate entries by taking the minimum value, making sure that the bond-features are included
            edge_index, edge_attr = coalesce(index=edge_index, value=edge_attr, m=pos.size(0), n = pos.size(0), op="min")
            
        if self._hparams["energy_preserving"]:
            pos.requires_grad_()

        pos_sde_traj = []
        pos_mean_traj = []
        
        chain = range(num_diffusion_timesteps)
        print(chain)
        for timestep in tqdm(reversed(chain), total=num_diffusion_timesteps):
            t = torch.full(size=(bs, ), fill_value=timestep, dtype=torch.long, device=pos.device)
            temb = self.timesteps_embedder[t][batch]
            out = self.model(
                x=x,
                t=temb,
                pos=pos,
                edge_index=edge_index,
                edge_attr=edge_attr,
                batch=batch,
            )
            if self._hparams["energy_preserving"]:
                grad_outputs: List[OptTensor] = [torch.ones_like(out)]
                out = torch.autograd.grad(
                    outputs=[out],
                    inputs=[pos],
                    grad_outputs=grad_outputs,
                    create_graph=True,
                    retain_graph=True,
                )[0]
        
            noise = torch.randn_like(pos)
            noise = noise - scatter_mean(noise, index=batch, dim=0, dim_size=bs)[batch]
        
            pos, pos_mean = self.sampler.update_fn(x=pos, score=out, t=t[batch], noise=noise)
            pos = pos - scatter_mean(pos, index=batch, dim=0, dim_size=bs)[batch]
            pos_mean = pos_mean - scatter_mean(pos_mean, index=batch, dim=0, dim_size=bs)[batch]

            if not self._hparams["fully_connected"]:
                edge_index = radius_graph(x=pos.detach(),
                                          r=self._hparams["cutoff"],
                                          batch=batch,
                                          max_num_neighbors=64
                                          )
                if self._hparams["use_bond_features"]:
                    # possibly combine the bond-edge-index with radius graph or fully-connected graph
                    # Note: This scenario is useful when learning the 3D coordinates only. 
                    # From an optimization perspective, atoms that are connected by topology should have certain distance values. 
                    # Since the atom types are fixed here, we know which molecule we want to generate a 3D configuration from, so the edge-index will help as inductive bias
                    edge_attr = torch.full(size=(edge_index.size(-1), ), fill_value=BOND_FEATURE_DIMS + 1, device=edge_index.device, dtype=torch.long)
                    # combine
                    edge_index = torch.cat([edge_index, bond_edge_index], dim=-1)
                    edge_attr =  torch.cat([edge_attr, bond_edge_attr], dim=0)
                    # coalesce, i.e. reduce and remove duplicate entries by taking the minimum value, making sure that the bond-features are included
                    edge_index, edge_attr = coalesce(index=edge_index, value=edge_attr, m=pos.size(0), n = pos.size(0), op="min")
            
                    
            if save_traj:
                pos_sde_traj.append(pos.detach())
                pos_mean_traj.append(pos_mean.detach())

        return pos, [pos_sde_traj, pos_mean_traj]

    def forward(self, batch: Batch, t: Tensor):
        node_feat: Tensor = batch.x
        pos: Tensor = batch.pos
        data_batch: Tensor = batch.batch
        ptr = batch.ptr
        bond_edge_index = batch.edge_index
        bond_edge_attr = batch.edge_attr

        bs = int(data_batch.max()) + 1
        
        if self._hparams["continuous"]:
            t_ = (t * (self.sde.N - 1)).long()
            timestep_embs = self.timesteps_embedder[t_][data_batch]
        else:
            timestep_embs = self.timesteps_embedder[t][data_batch]
            
        # center the true point cloud
        pos_centered = pos - scatter_mean(pos, index=data_batch, dim=0, dim_size=bs)[data_batch]

        # sample 0-COM noise
        noise = torch.randn_like(pos)
        noise = noise - scatter_mean(noise, index=data_batch, dim=0, dim_size=bs)[data_batch]
        
        # get mean and std of pos_t | pos_0
        mean, std = self.sde.marginal_prob(x=pos_centered, t=t[data_batch])
        
        # perturb
        pos_perturbed = mean + std * noise
        
        if self._hparams["fully_connected"]:
            edge_index = batch.edge_index_fc
            if edge_index is None:
                # fully-connected graphs
                batch_num_nodes = torch.bincount(data_batch)
                edge_index_list = []
                for offset, n in zip(ptr.cpu().tolist(), batch_num_nodes.cpu().tolist()):
                    row = torch.arange(n, dtype=torch.long)
                    col = torch.arange(n, dtype=torch.long)
                    row = row.view(-1, 1).repeat(1, n).view(-1)
                    col = col.repeat(n)
                    edge_index = torch.stack([row, col], dim=0)
                    mask = edge_index[0] != edge_index[1]
                    edge_index = edge_index[:, mask]
                    edge_index += offset
                    edge_index_list.append(edge_index)
                edge_index = torch.concat(edge_index_list, dim=-1).to(node_feat.device)     
        else:
            edge_index = radius_graph(x=pos_perturbed, r=self._hparams["cutoff"], batch=data_batch, max_num_neighbors=64)
        
        if self._hparams["use_bond_features"]:
            # possibly combine the bond-edge-index with radius graph or fully-connected graph
            # Note: This scenario is useful when learning the 3D coordinates only. 
            # From an optimization perspective, atoms that are connected by topology should have certain distance values. 
            # Since the atom types are fixed here, we know which molecule we want to generate a 3D configuration from, so the edge-index will help as inductive bias
            edge_attr = torch.full(size=(edge_index.size(-1), ), fill_value=BOND_FEATURE_DIMS + 1, device=edge_index.device, dtype=torch.long)
            # combine
            edge_index = torch.cat([edge_index, bond_edge_index], dim=-1)
            edge_attr =  torch.cat([edge_attr, bond_edge_attr], dim=0)
            # coalesce, i.e. reduce and remove duplicate entries by taking the minimum value, making sure that the bond-features are included
            edge_index, edge_attr = coalesce(index=edge_index, value=edge_attr, m=pos.size(0), n = pos.size(0), op="min")
            
        if self._hparams["energy_preserving"]:
            pos_perturbed.requires_grad = True

        out = self.model(
            x=node_feat,
            t=timestep_embs,
            pos=pos_perturbed,
            edge_index=edge_index,
            edge_attr=edge_attr if self._hparams["use_bond_features"] else None,
            batch=data_batch,
        )

        if self._hparams["energy_preserving"]:
            grad_outputs: List[OptTensor] = [torch.ones_like(out)]
            energy_norm = torch.pow(out, 2).sum(-1).mean(0)
            out = torch.autograd.grad(
                outputs=[out],
                inputs=[pos_perturbed],
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True,
            )[0]
        else:
            energy_norm = 0.0

        # center the predictions as ground truth noise is also centered
        out_mean = scatter_mean(out, index=data_batch, dim=0, dim_size=bs)
        out = out - out_mean[data_batch]

        out_dict = {"pred_noise": out, "true_noise": noise, "energy_norm": energy_norm}
        return out_dict

    def training_step(self, batch, batch_idx, debug: bool = False):
        batch_size = int(batch.batch.max()) + 1
        
        if self._hparams["continuous"]:
            t = torch.rand(size=(batch_size, ), dtype=batch.pos.dtype, device=batch.pos.device)
            t = t * (self.sde.T - self._hparams["eps_min"]) + self._hparams["eps_min"]
        else:
            # ToDo: Check the discrete state t=0
            t = torch.randint(low=0, high=self._hparams['num_diffusion_timesteps'],
                              size=(batch_size,), 
                              dtype=torch.long, device=batch.x.device)
            
        out_dict = self(batch=batch, t=t)
        loss = torch.pow(out_dict["pred_noise"] - out_dict["true_noise"], 2).sum(-1)
        loss = scatter_mean(loss, index=batch.batch, dim=0, dim_size=batch_size)
        loss = torch.mean(loss, dim=0) 
        loss = loss + 1e-2 * out_dict["energy_norm"]
        self.log(
            "train/loss",
            loss,
            on_step=True,
            batch_size=batch_size,
        )
        
        if not self._hparams["continuous"]:
            t = t.float() / self._hparams["num_diffusion_timesteps"]
            
        time_mean = torch.mean(t)
        time_var = torch.std(t).pow(2)
        self.log(
            "train/time_mean",
            time_mean,
            on_step=True,
            batch_size=batch_size,
        )
        self.log(
            "train/time_var",
            time_var,
            on_step=True,
            batch_size=batch_size,
        )
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        if self._hparams["energy_preserving"]:
            torch.set_grad_enabled(True)

        batch_size = int(batch.batch.max()) + 1
        if self._hparams["continuous"]:
            t = torch.rand(size=(batch_size, ), dtype=batch.pos.dtype, device=batch.pos.device)
            t = t * (self.sde.T - self._hparams["eps_min"]) + self._hparams["eps_min"]
        else:
            # ToDo: Check the discrete state t=0
            t = torch.randint(low=0, high=self._hparams['num_diffusion_timesteps'],
                              size=(batch_size,), 
                              dtype=torch.long, device=batch.x.device)
        out_dict = self(batch=batch, t=t)

        loss = torch.pow(out_dict["pred_noise"] - out_dict["true_noise"], 2).sum(-1)
        loss = scatter_mean(loss, index=batch.batch, dim=0, dim_size=batch_size)
        loss = torch.mean(loss, dim=0)
        loss = loss + 1e-2 * out_dict["energy_norm"]
        
        self.log(
            "val/loss",
            loss,
            batch_size=batch_size,
            sync_dist=self._hparams["gpus"] > 1,
        )
        
        if not self._hparams["continuous"]:
            t = t.float() / self._hparams["num_diffusion_timesteps"]
            
        time_mean = torch.mean(t)
        time_var = torch.std(t).pow(2)
        self.log(
            "val/time_mean",
            time_mean,
            batch_size=batch_size,
            sync_dist=self._hparams["gpus"] > 1,
        )
        self.log(
            "val/time_var",
            time_var,
            batch_size=batch_size,
            sync_dist=self._hparams["gpus"] > 1,
        )
        
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self._hparams["lr"])
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            patience=self._hparams["patience"],
            cooldown=self._hparams["cooldown"],
            factor=self._hparams["factor"],
        )
        scheduler = {
            "scheduler": lr_scheduler,
            "interval": "epoch",
            "frequency": self._hparams["frequency"],
            "monitor": "val/loss",
        }
        return [optimizer], [scheduler]


if __name__ == "__main__":
    from geom.data import GeomDataModule
    from geom.hparams import add_arguments

    parser = ArgumentParser()
    parser = add_arguments(parser)
    hparams = parser.parse_args()
    
    if not os.path.exists(hparams.save_dir):
        os.makedirs(hparams.save_dir)

    if not os.path.isdir(hparams.save_dir + f"/run{hparams.id}/"):
        print("Creating directory")
        os.mkdir(hparams.save_dir + f"/run{hparams.id}/")
    print(f"Starting Run {hparams.id}")
    checkpoint_callback = ModelCheckpoint(
        dirpath=hparams.save_dir + f"/run{hparams.id}/",
        save_top_k=1,
        monitor="val/loss",
        save_last=True,
    )
    lr_logger = LearningRateMonitor()
    tb_logger = TensorBoardLogger(
        hparams.save_dir + f"/run{hparams.id}/", default_hp_metric=False
    )

    model = Trainer(hparams=hparams.__dict__)

    print(f"Loading {hparams.dataset} Datamodule.")
    datamodule = GeomDataModule(
        batch_size=hparams.batch_size,
        num_workers=hparams.num_workers,
        dataset=hparams.dataset,
        env_in_init=True,
        shuffle_train=True,
        # subset_frac=hparams.subset_frac,
        max_num_conformers=hparams.max_num_conformers,
        transform=MolFeaturization(order=3),
        pin_memory=True,
        persistent_workers=True
    )

    strategy = (
        pl.strategies.DDPStrategy(find_unused_parameters=False)
        if hparams.gpus > 1
        else None
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=hparams.gpus,
        strategy=strategy,
        logger=tb_logger,
        enable_checkpointing=True,
        accumulate_grad_batches=hparams.accum_batch,
        val_check_interval=hparams.eval_freq,
        gradient_clip_val=hparams.grad_clip_val,
        callbacks=[
            lr_logger,
            checkpoint_callback,
            TQDMProgressBar(refresh_rate=5),
            ModelSummary(max_depth=2),
        ],
        precision=hparams.precision,
        num_sanity_val_steps=2,
        max_epochs=hparams.num_epochs,
        detect_anomaly=hparams.detect_anomaly,
    )

    pl.seed_everything(seed=0, workers=hparams.gpus > 1)

    trainer.fit(
        model=model,
        datamodule=datamodule,
        ckpt_path=hparams.load_ckpt if hparams.load_ckpt != "" else None,
    )
