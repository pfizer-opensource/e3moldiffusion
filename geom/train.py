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
from e3moldiffusion.sde import VPSDE, VPAncestralSamplingPredictor, get_timestep_embedding
from e3moldiffusion.gnn import EQGATEncoder

from torch import Tensor
from torch_geometric.data import Batch
from torch_geometric.typing import OptTensor
from torch_cluster import radius_graph
from torch_scatter import scatter_mean
from tqdm import tqdm

logging.getLogger("lightning").setLevel(logging.WARNING)

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
        )

        timesteps = torch.arange(hparams["num_diffusion_timesteps"], dtype=torch.long)
        timesteps_embedding = get_timestep_embedding(
            timesteps=timesteps, embedding_dim=hparams["tdim"]
        ).to(torch.float32)

        self.register_buffer("timesteps", tensor=timesteps)
        self.register_buffer("timesteps_embedding", tensor=timesteps_embedding)

        self.sde = VPSDE(beta_min=hparams["beta_min"], beta_max=hparams["beta_max"],
                         N=hparams["num_diffusion_timesteps"],
                         scaled_reverse_posterior_sigma=False
                         )
        
        self.sampler = VPAncestralSamplingPredictor(sde=self.sde)
        

    def reverse_sampling(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: OptTensor = None,
        batch: OptTensor = None,
        num_diffusion_timesteps: Optional[int] = None,
        save_traj: bool = False,
    ) -> Tuple[Tensor, List]:
        if num_diffusion_timesteps is None:
            num_diffusion_timesteps = self._hparams["num_diffusion_timesteps"]

        if batch is None:
            batch = torch.zeros(len(x), device=x.device, dtype=torch.long)

        bs = int(batch.max()) + 1
        num_graphs = int(batch.max()) + 1

        pos = torch.randn(x.size(0), 3, device=x.device, dtype=self.timesteps_embedding.dtype)
        if self._hparams["energy_preserving"]:
            pos.requires_grad_()

        pos = pos - scatter_mean(pos, dim=0, index=batch, dim_size=bs)[batch]

        pos_sde_traj = []
        pos_mean_traj = []
        
        chain = range(
            self._hparams["num_diffusion_timesteps"] - num_diffusion_timesteps,
            self._hparams["num_diffusion_timesteps"],
        )
        print(chain)

        for t_ in tqdm(reversed(chain), total=num_diffusion_timesteps):
            t = torch.full(
                size=(num_graphs,), fill_value=t_, dtype=torch.long, device=x.device
            )
            temb = self.timesteps_embedding[t][batch]
            
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
            
            pos, pos_mean = self.sampler.update_fn(x=pos, score=out, t=t, noise=noise)
          
            if save_traj:
                pos_sde_traj.append(pos.detach())
                pos_mean_traj.append(pos_mean.detach())

        return pos, [pos_sde_traj, pos_mean_traj]

    def forward(self, batch: Batch, t: OptTensor = None):
        node_feat: Tensor = batch.x
        pos: Tensor = batch.pos
        data_batch: Tensor = batch.batch
        ptr = batch.ptr

        if t is None:
            t = torch.zeros(
                size=(node_feat.size(0),), device=node_feat.device, dtype=pos.dtype
            )
        if data_batch is None:
            data_batch = torch.zeros(
                size=(node_feat.size(0),), device=node_feat.device, dtype=torch.long
            )
        
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
            edge_index = radius_graph(x=pos, r=self._hparams["cutoff"], batch=data_batch, max_num_neighbors=64)

        bs = int(data_batch.max()) + 1
        
        timestep_embs = t * (self._hparams["num_diffusion_timesteps"] - 1)
        timestep_embs = self.timesteps_embedding[timestep_embs.long()][data_batch]

        pos_centered = pos - scatter_mean(pos, index=data_batch, dim=0, dim_size=bs)[data_batch]

        # sample COM noise
        noise = torch.randn_like(pos)
        noise = noise - scatter_mean(noise, index=data_batch, dim=0, dim_size=bs)[data_batch]
        
        # get mean and std of pos_t | pos_0
        mean, std = self.sde.marginal_prob(x=pos_centered, t=t[data_batch])
        
        # perturb
        pos_perturbed = mean + std * noise
        
        if self._hparams["energy_preserving"]:
            pos_perturbed.requires_grad = True

        out = self.model(
            x=node_feat,
            t=timestep_embs,
            pos=pos_perturbed,
            edge_index=edge_index,
            edge_attr=None,
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

        out_mean = scatter_mean(out, index=data_batch, dim=0, dim_size=bs)
        out = out - out_mean[data_batch]

        out_dict = {"pred_noise": out, "true_noise": noise, "energy_norm": energy_norm}

        return out_dict

    def training_step(self, batch, batch_idx, debug: bool = False):
        batch_size = int(batch.batch.max()) + 1
        t = torch.rand(size=(batch_size, ), dtype=batch.pos.dtype, device=batch.pos.device)
        t = t * (self.sde.T - self._hparams["eps_min"]) + self._hparams["eps_min"]
        out_dict = self(batch=batch, t=t)
        loss = torch.pow(out_dict["pred_noise"] - out_dict["true_noise"], 2).sum(-1)
        loss = scatter_mean(loss, index=batch.batch, dim=0, dim_size=batch_size)
        loss = torch.mean(loss, dim=0) 
        loss = loss + out_dict["energy_norm"]
        self.log(
            "train/loss",
            loss,
            on_step=True,
            batch_size=batch_size,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        if self._hparams["energy_preserving"]:
            torch.set_grad_enabled(True)

        batch_size = int(batch.batch.max()) + 1
        t = torch.rand(size=(batch_size, ), dtype=batch.pos.dtype, device=batch.pos.device)
        t = t * (self.sde.T - self._hparams["eps_min"]) + self._hparams["eps_min"]
        out_dict = self(batch=batch, t=t)

        loss = torch.pow(out_dict["pred_noise"] - out_dict["true_noise"], 2).sum(-1)
        loss = scatter_mean(loss, index=batch.batch, dim=0, dim_size=batch_size)
        loss = torch.mean(loss, dim=0)
        loss = loss + out_dict["energy_norm"]
        
        self.log(
            "val/loss",
            loss,
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
        subset_frac=hparams.subset_frac,
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

    pl.seed_everything(seed=0)

    trainer.fit(
        model=model,
        datamodule=datamodule,
        ckpt_path=hparams.load_ckpt if hparams.load_ckpt != "" else None,
    )
