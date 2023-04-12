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
from e3moldiffusion.molfeat import get_bond_feature_dims, atom_type_config
from e3moldiffusion.sde import VPSDE, VPAncestralSamplingPredictor, DiscreteDDPM
from e3moldiffusion.coords import ScoreModel

from torch import Tensor
from torch_geometric.data import Batch
from torch_geometric.typing import OptTensor
from torch_geometric.utils import dense_to_sparse
from torch_sparse import coalesce
from torch_scatter import scatter_mean
from tqdm import tqdm

logging.getLogger("lightning").setLevel(logging.WARNING)


def get_num_atom_types_geom(dataset: str):
    assert dataset in ["qm9", "drugs"]
    return len(atom_type_config(dataset=dataset))

BOND_FEATURE_DIMS = get_bond_feature_dims()[0]

class Trainer(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.hparams.num_atom_types = get_num_atom_types_geom(dataset=hparams["dataset"])
        self.model = ScoreModel(
            num_atom_types=self.hparams.num_atom_types,
            hn_dim=(hparams["sdim"], hparams["vdim"]),
            edge_dim=hparams["edim"],
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
    
    def coalesce_edges(self, edge_index, bond_edge_index, bond_edge_attr, n):
        # possibly combine the bond-edge-index with radius graph or fully-connected graph
        # Note: This scenario is useful when learning the 3D coordinates only. 
        # From an optimization perspective, atoms that are connected by topology should have certain distance values. 
        # Since the atom types are fixed here, we know which molecule we want to generate a 3D configuration from, so the edge-index will help as inductive bias
        edge_attr = torch.full(size=(edge_index.size(-1), ), fill_value=BOND_FEATURE_DIMS + 1, device=edge_index.device, dtype=torch.long)
        # combine
        edge_index = torch.cat([edge_index, bond_edge_index], dim=-1)
        edge_attr =  torch.cat([edge_attr, bond_edge_attr], dim=0)
        # coalesce, i.e. reduce and remove duplicate entries by taking the minimum value, making sure that the bond-features are included
        edge_index, edge_attr = coalesce(index=edge_index, value=edge_attr, m=n, n=n, op="min")
        return edge_index, edge_attr
        
    def reverse_sampling(
        self,
        x: Tensor,
        batch: OptTensor = None,
        num_diffusion_timesteps: Optional[int] = None,
        save_traj: bool = False,
        bond_edge_index: OptTensor = None,
        bond_edge_attr: OptTensor = None,
        verbose: bool = True
    ) -> Tuple[Tensor, List]:
                
        if num_diffusion_timesteps is None:
            num_diffusion_timesteps = self.hparams.num_diffusion_timesteps

        if batch is None:
            batch = torch.zeros(len(x), device=x.device, dtype=torch.long)

        if self.hparams.use_bond_features:
            assert bond_edge_index is not None
            assert bond_edge_attr is not None
            
        bs = int(batch.max()) + 1
                
        # initialiaze the 0-mean point cloud from N(0, I)
        pos = torch.randn(x.size(0), 3,
                          device=x.device,
                          dtype=torch.get_default_dtype())
        pos = pos - scatter_mean(pos, dim=0, index=batch, dim_size=bs)[batch]
        
        edge_index_local, edge_attr_local = bond_edge_index, bond_edge_attr
        
        edge_index_global = torch.eq(batch.unsqueeze(0), batch.unsqueeze(-1)).int().fill_diagonal_(0)
        edge_index_global, _ = dense_to_sparse(edge_index_global)

        if self.hparams.use_bond_features:    
            edge_index_global, edge_attr_global = self.coalesce_edges(edge_index=edge_index_global,
                                                                      bond_edge_index=bond_edge_index,
                                                                      bond_edge_attr=bond_edge_attr,
                                                                      n=pos.size(0))

        pos_sde_traj = []
        pos_mean_traj = []
        
        xohe = F.one_hot(x, num_classes=self.hparams.num_atom_types).float()
        edge_attr_local = F.one_hot(edge_attr_local, num_classes=BOND_FEATURE_DIMS + 1).float()
        edge_attr_global = F.one_hot(edge_attr_global, num_classes=BOND_FEATURE_DIMS + 1).float()


        chain = range(num_diffusion_timesteps)
        if verbose:
            print(chain)
        iterator = tqdm(reversed(chain), total=num_diffusion_timesteps) if verbose else reversed(chain)
        for timestep in iterator:
            t = torch.full(size=(bs, ), fill_value=timestep, dtype=torch.long, device=pos.device)
            temb = t / self.hparams.num_diffusion_timesteps
            temb = temb.unsqueeze(dim=1)
            temb = temb.index_select(dim=0, index=batch)
            
            out = self.model(
                x=xohe,
                t=temb,
                pos=pos,
                edge_index_local=edge_index_local,
                edge_index_global=edge_index_global,
                edge_attr_local=edge_attr_local if self.hparams.use_bond_features else None,
                edge_attr_global=edge_attr_global if self.hparams.use_bond_features else None,
                batch=batch,
            )
            noise = torch.randn_like(pos)
            noise = noise - scatter_mean(noise, index=batch, dim=0, dim_size=bs)[batch]
        
            pos, pos_mean = self.sampler.update_fn(x=pos, score=out, t=t[batch], noise=noise)
            pos = pos - scatter_mean(pos, index=batch, dim=0, dim_size=bs)[batch]
            pos_mean = pos_mean - scatter_mean(pos_mean, index=batch, dim=0, dim_size=bs)[batch]

            if save_traj:
                pos_sde_traj.append(pos.detach())
                pos_mean_traj.append(pos_mean.detach())

        return pos, [pos_sde_traj, pos_mean_traj]

    def forward(self, batch: Batch, t: Tensor):
        node_feat: Tensor = batch.xgeom
        pos: Tensor = batch.pos
        data_batch: Tensor = batch.batch
        bond_edge_index = batch.edge_index
        bond_edge_attr = batch.edge_attr

        assert pos.shape[0] == t.shape[0]

        bs = int(data_batch.max()) + 1
        
        if not self.hparams.continuous:
            temb = t.float() / self.hparams.num_diffusion_timesteps
            temb = temb.clamp(min=self.hparams.eps_min)
        else:
            temb = t
            
        temb = temb.unsqueeze(dim=1)
        
        # center the true point cloud
        pos_centered = pos - scatter_mean(pos, index=data_batch, dim=0, dim_size=bs)[data_batch]

        # sample 0-COM noise
        noise = torch.randn_like(pos)
        noise = noise - scatter_mean(noise, index=data_batch, dim=0, dim_size=bs)[data_batch]
        
        # get mean and std of pos_t | pos_0
        mean, std = self.sde.marginal_prob(x=pos_centered, t=t[data_batch])
        
        # perturb
        pos_perturbed = mean + std * noise
        
        edge_index_local, edge_attr_local = bond_edge_index, bond_edge_attr
        if not hasattr(batch, "edge_index_fc"):
            edge_index_global = torch.eq(batch.batch.unsqueeze(0), batch.batch.unsqueeze(-1)).int().fill_diagonal_(0)
            edge_index_global, _ = dense_to_sparse(edge_index_global)
        else:
            edge_index_global = batch.edge_index_fc

        if self.hparams.use_bond_features:
            edge_index_global, edge_attr_global = self.coalesce_edges(edge_index=edge_index_global,
                                                                      bond_edge_index=bond_edge_index, bond_edge_attr=bond_edge_attr,
                                                                      n=pos.size(0))
        
        xohe = F.one_hot(node_feat, num_classes=self.hparams.num_atom_types).float()
        edge_attr_local = F.one_hot(edge_attr_local, num_classes=BOND_FEATURE_DIMS + 1).float()
        edge_attr_global = F.one_hot(edge_attr_global, num_classes=BOND_FEATURE_DIMS + 1).float()
        
        out = self.model(
            x=xohe,
            t=temb,
            pos=pos_perturbed,
            edge_index_local=edge_index_local,
            edge_index_global=edge_index_global,
            edge_attr_local=edge_attr_local if self.hparams.use_bond_features else None,
            edge_attr_global=edge_attr_global if self.hparams.use_bond_features else None,
            batch=data_batch,
        )

        # center the predictions as ground truth noise is also centered
        out_mean = scatter_mean(out, index=data_batch, dim=0, dim_size=bs)
        out = out - out_mean[data_batch]

        out_dict = {"pred_noise": out, "true_noise": noise}
        return out_dict
    
    def step_fnc(self, batch, batch_idx, stage):         
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
        loss = torch.pow(out_dict["pred_noise"] - out_dict["true_noise"], 2).sum(-1)
        loss = scatter_mean(loss, index=batch.batch, dim=0, dim_size=batch_size)
        loss = torch.mean(loss, dim=0)
        
        if stage == "val":
            sync_dist =  self.hparams.gpus > 1
        else:
            sync_dist = False
            
        self.log(
            f"{stage}/coords_loss",
            loss,
            on_step=True,
            batch_size=batch_size,
            sync_dist=sync_dist
        )
        return loss
        
    def training_step(self, batch, batch_idx):
        return self.step_fnc(batch=batch, batch_idx=batch_idx, stage="train")
    
    def validation_step(self, batch, batch_idx):
        return self.step_fnc(batch=batch, batch_idx=batch_idx, stage="val")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            patience=self.hparams.patience,
            cooldown=self.hparams.cooldown,
            factor=self.hparams.factor,
        )
        # ToDo ExponentialLR 
        scheduler = {
            "scheduler": lr_scheduler,
            "interval": "epoch",
            "frequency": self.hparams.frequency,
            "monitor": "val/coords_loss",
        }
        return [optimizer], [scheduler]

if __name__ == "__main__":
    from geom.data import GeomDataModule
    from geom.hparams_coords import add_arguments

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
        monitor="val/coords_loss",
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
        transform_args = {"create_bond_graph": True,
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
        accelerator="gpu" if hparams.gpus else "cpu",
        devices=hparams.gpus if hparams.gpus else None,
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
