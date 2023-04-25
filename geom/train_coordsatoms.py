import logging
import os
from argparse import ArgumentParser
from typing import List, Optional, Tuple

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
    TQDMProgressBar,
)
import torch.nn.functional as F 

from pytorch_lightning.loggers import TensorBoardLogger
from e3moldiffusion.molfeat import atom_type_config
from e3moldiffusion.sde import VPSDE, VPAncestralSamplingPredictor, DiscreteDDPM
from e3moldiffusion.coordsatoms import ScoreModel

from torch import Tensor
from torch_geometric.data import Batch
from torch_geometric.typing import OptTensor
from torch_geometric.nn import radius_graph
from torch_geometric.utils import dense_to_sparse, sort_edge_index
from torch_scatter import scatter_mean
from tqdm import tqdm

logging.getLogger("lightning").setLevel(logging.WARNING)

def get_num_atom_types_geom(dataset: str):
    assert dataset in ["qm9", "drugs"]
    return len(atom_type_config(dataset=dataset))


def zero_mean(x: Tensor, batch: Tensor, dim_size: int, dim=0):
    out = x - scatter_mean(x, index=batch, dim=dim, dim_size=dim_size)[batch]
    return out

def assert_zero_mean(x: Tensor, batch: Tensor, dim_size: int, dim=0, eps: float = 1e-6):
    out = scatter_mean(x, index=batch, dim=dim, dim_size=dim_size).mean()
    return abs(out) < eps


class Trainer(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.hparams.num_atom_types = get_num_atom_types_geom(dataset=hparams["dataset"])
        self.model = ScoreModel(
            num_atom_types=self.hparams.num_atom_types,
            hn_dim=(hparams["sdim"], hparams["vdim"]),
            cutoff_local=hparams["cutoff_local"],
            cutoff_global=hparams["cutoff_global"],
            rbf_dim=hparams["rbf_dim"],
            num_layers=hparams["num_layers"],
            use_norm=not hparams["omit_norm"],
            use_cross_product=not hparams["omit_cross_product"],
            vector_aggr=hparams["vector_aggr"],
            fully_connected=hparams["fully_connected"],
            local_global_model=hparams["local_global_model"],
        )

        if hparams["continuous"]:
            # todo: double check continuous time
            raise NotImplementedError
            self.sde = VPSDE(beta_min=hparams["beta_min"],
                             beta_max=hparams["beta_max"],
                             N=hparams["num_diffusion_timesteps"],
                             scaled_reverse_posterior_sigma=True)
        else:
            self.sde = DiscreteDDPM(beta_min=hparams["beta_min"],
                                    beta_max=hparams["beta_max"],
                                    N=hparams["num_diffusion_timesteps"],
                                    scaled_reverse_posterior_sigma=False,
                                    schedule=hparams["schedule"])
            
        self.sampler = VPAncestralSamplingPredictor(sde=self.sde)
     
    
    def reverse_sampling(
        self,
        num_graphs: int,
        device: torch.device,
        empirical_distribution_num_nodes: Tensor,
        verbose: bool = False,
        save_traj: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, List]:
        
        batch_num_nodes = torch.multinomial(input=empirical_distribution_num_nodes,
                                            num_samples=num_graphs, replacement=True).to(device)
        batch_num_nodes = batch_num_nodes.clamp(min=1)
        batch = torch.arange(num_graphs, device=device).repeat_interleave(batch_num_nodes, dim=0)
        bs = int(batch.max()) + 1
        
        # initialiaze the 0-mean point cloud from N(0, I)
        pos = torch.randn(len(batch), 3,
                          device=device,
                          dtype=torch.get_default_dtype()
                          )
        pos = zero_mean(pos, batch=batch, dim_size=bs, dim=0)
        
        # initialize the atom-types 
        atom_types = torch.randn(pos.size(0), self.hparams.num_atom_types, device=device)
        
        edge_index_local = radius_graph(x=pos,
                                        r=self.hparams.cutoff_local,
                                        batch=batch, 
                                        max_num_neighbors=self.hparams.max_num_neighbors)
        
        edge_index_global = torch.eq(batch.unsqueeze(0), batch.unsqueeze(-1)).int().fill_diagonal_(0)
        edge_index_global, _ = dense_to_sparse(edge_index_global)

        pos_traj = []
        atom_type_traj = []
        atom_type_ohe_traj = []
        chain = range(self.hparams.num_diffusion_timesteps)
    
        if verbose:
            print(chain)
        iterator = tqdm(reversed(chain), total=len(chain)) if verbose else reversed(chain)
        for timestep in iterator:
            t = torch.full(size=(bs, ), fill_value=timestep, dtype=torch.long, device=pos.device)
            temb = t / self.hparams.num_diffusion_timesteps
            temb = temb.unsqueeze(dim=1)
            out = self.model(
                x=atom_types,
                t=temb,
                pos=pos,
                edge_index_local=edge_index_local,
                edge_index_global=edge_index_global,
                edge_attr_local=None,
                edge_attr_global=None,
                batch=batch,
            )
             
            score_coords = out["score_coords"]
            score_atoms = out["score_atoms"]
            
            score_coords = zero_mean(score_coords, batch=batch, dim_size=bs, dim=0)
            noise_coords = torch.randn_like(pos)
            noise_coords = zero_mean(noise_coords, batch=batch, dim_size=bs, dim=0)
        
            pos, _ = self.sampler.update_fn(x=pos, score=score_coords, t=t[batch], noise=noise_coords)
            pos = zero_mean(pos, batch=batch, dim_size=bs, dim=0)
            
            noise_atoms = torch.randn_like(atom_types)
            atom_types, _ = self.sampler.update_fn(x=atom_types, score=score_atoms, t=t[batch], noise=noise_atoms)
            
            if not self.hparams.fully_connected:
                edge_index_local = radius_graph(x=pos.detach(),
                                                r=self.hparams.cutoff_local,
                                                batch=batch, 
                                                max_num_neighbors=self.hparams.max_num_neighbors)
            
            atom_integer = torch.argmax(atom_types, dim=-1)
            
            if save_traj:
                pos_traj.append(pos.detach())
                atom_type_traj.append(atom_types.detach())
                atom_type_ohe_traj.append(atom_integer)
                
        return pos, atom_types, atom_integer, batch_num_nodes, [pos_traj, atom_type_traj, atom_type_ohe_traj]
    
    def forward(self, batch: Batch, t: Tensor):
        node_feat: Tensor = batch.xgeom
        pos: Tensor = batch.pos
        data_batch: Tensor = batch.batch
        bs = int(data_batch.max()) + 1
                
        if not self.hparams.continuous:
            temb = t.float() / self.hparams.num_diffusion_timesteps
            temb = temb.clamp(min=self.hparams.eps_min)
        else:
            temb = t
            
        temb = temb.unsqueeze(dim=1)
        
        # Coords: point cloud in R^3
        # sample noise for coords and recenter
        noise_coords_true = torch.randn_like(pos)
        noise_coords_true = zero_mean(noise_coords_true, batch=data_batch, dim_size=bs, dim=0)
        # center the true point cloud
        pos_centered = zero_mean(pos, data_batch, dim=0, dim_size=bs)
        # get signal and noise coefficients for coords
        mean_coords, std_coords = self.sde.marginal_prob(x=pos_centered, t=t[data_batch])
        # perturb coords
        pos_perturbed = mean_coords + std_coords * noise_coords_true
        
        # one-hot-encode
        atom_types = F.one_hot(node_feat, num_classes=self.hparams.num_atom_types).float()
        atom_types = 0.25 * atom_types
        # sample noise for OHEs in {0, 1}^NUM_CLASSES
        noise_atom_types = torch.randn_like(atom_types)
        mean_atom_types, std_atom_types = self.sde.marginal_prob(x=atom_types, t=t[data_batch])
        # perturb OHEs
        atom_types_perturbed = mean_atom_types + std_atom_types * noise_atom_types
        
        edge_index_local = radius_graph(x=pos_perturbed,
                                        r=self.hparams.cutoff_local,
                                        batch=data_batch, 
                                        max_num_neighbors=self.hparams.max_num_neighbors)
        
        if not hasattr(batch, "edge_index_fc"):
            edge_index_global = torch.eq(batch.batch.unsqueeze(0), batch.batch.unsqueeze(-1)).int().fill_diagonal_(0)
            edge_index_global, _ = dense_to_sparse(edge_index_global)
        else:
            edge_index_global = batch.edge_index_fc

        out = self.model(
            x=atom_types_perturbed,
            t=temb,
            pos=pos_perturbed,
            edge_index_local=edge_index_local,
            edge_index_global=edge_index_global,
            edge_attr_local=None,
            edge_attr_global=None,
            batch=data_batch,
        )

        noise_atoms_pred = out["score_atoms"]
        noise_coords_pred = out["score_coords"]
        noise_coords_pred = zero_mean(noise_coords_pred, batch=data_batch, dim_size=bs, dim=0)
        
        out = {
            "noise_coords_pred": noise_coords_pred,
            "noise_coords_true": noise_coords_true,
            "noise_atoms_pred": noise_atoms_pred,
            "noise_atoms_true": noise_atom_types,
            "true_class": node_feat,
        }
        
        return out
    
    def step_fnc(self, batch, batch_idx, stage: str):
        batch_size = int(batch.batch.max()) + 1
        if self.hparams.continuous:
            t = torch.rand(size=(batch_size, ), dtype=batch.pos.dtype, device=batch.pos.device)
            t = t * (self.sde.T - self.hparams.eps_min) + self.hparams.eps_min
        else:
            # ToDo: Check the discrete state t=0
            t = torch.randint(low=0, high=self.hparams.num_diffusion_timesteps,
                              size=(batch_size,), 
                              dtype=torch.long, device=batch.x.device)
            
        out_dict = self(batch=batch, t=t)
        
        coords_loss = torch.pow(
            out_dict["noise_coords_pred"] - out_dict["noise_coords_true"], 2
        ).sum(-1)
        coords_loss = scatter_mean(
            coords_loss, index=batch.batch, dim=0, dim_size=batch_size
        )
        coords_loss = torch.mean(coords_loss, dim=0)
        
        atoms_loss = torch.pow(
            out_dict["noise_atoms_pred"] - out_dict["noise_atoms_true"], 2
        ).mean(-1) 
        atoms_loss = scatter_mean(
            atoms_loss, index=batch.batch, dim=0, dim_size=batch_size
        )
        atoms_loss = torch.mean(atoms_loss, dim=0)

        loss = coords_loss + atoms_loss

        self.log(
            f"{stage}/loss",
            loss,
            on_step=True,
            batch_size=batch_size,
        )

        self.log(
            f"{stage}/coords_loss",
            coords_loss,
            on_step=True,
            batch_size=batch_size,
        )

        self.log(
            f"{stage}/atoms_loss",
            atoms_loss,
            on_step=True,
            batch_size=batch_size,
        )
        return loss

    def training_step(self, batch, batch_idx):
        return self.step_fnc(batch=batch, batch_idx=batch_idx, stage="train")

    def validation_step(self, batch, batch_idx):
        return self.step_fnc(batch=batch, batch_idx=batch_idx, stage="val")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            patience=self.hparams.patience,
            cooldown=self.hparams.cooldown,
            factor=self.hparams.factor,
        )
        scheduler = {
            "scheduler": lr_scheduler,
            "interval": "epoch",
            "frequency": 1,
            "monitor": "val/loss",
        }
        return [optimizer], [scheduler]
    

if __name__ == "__main__":
    from geom.data import GeomDataModule
    from geom.hparams_coordsatoms import add_arguments

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
        max_num_conformers=hparams.max_num_conformers,
        pin_memory=True,
        persistent_workers=True,
        transform_args = {"create_bond_graph": False,
                          "save_smiles": False,
                          "fully_connected_edge_index": False
                          }
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
        max_time=hparams.max_time
    )

    pl.seed_everything(seed=0, workers=hparams.gpus > 1)

    trainer.fit(
        model=model,
        datamodule=datamodule,
        ckpt_path=hparams.load_ckpt if hparams.load_ckpt != "" else None,
    )
