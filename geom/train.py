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
from pytorch_lightning.loggers import TensorBoardLogger
from e3moldiffusion.molfeat import get_bond_feature_dims
from e3moldiffusion.sde import VPSDE, VPAncestralSamplingPredictor, get_timestep_embedding, DiscreteDDPM
from e3moldiffusion.gnn import ScoreModelCoords

from torch import Tensor
from torch_geometric.data import Batch
from torch_geometric.typing import OptTensor
from torch_geometric.utils import dense_to_sparse, to_dense_adj, sort_edge_index
from torch_cluster import radius_graph
from torch_sparse import coalesce
from torch_scatter import scatter_mean
from tqdm import tqdm

logging.getLogger("lightning").setLevel(logging.WARNING)

BOND_FEATURE_DIMS = get_bond_feature_dims()[0]

def binarize(x):
        return torch.where(x > 0, torch.ones_like(x), torch.zeros_like(x))


def triple_order_edges(bond_edge_index: Tensor,
                       bond_edge_attr: Tensor,
                       batch: Tensor,
                       check_coalesce: bool = False) -> Tuple[Tensor, Tensor]:
    
    # store in big block-triangular matrix
    dense_adj = torch.zeros(size=(len(batch), len(batch)), dtype=torch.float32, device=bond_edge_index.device)
    dense_adj[bond_edge_index[0, :], bond_edge_index[1, :]] = 1.0
    no_self_loop = torch.ones(dense_adj.size(0), dense_adj.size(1)).fill_diagonal_(0.0)
    adj1 = dense_adj * no_self_loop
    adj2 = binarize(adj1 @ adj1) * no_self_loop
    new_ids_2 = (1.0 - adj1) * adj2
    
    edge_index_two_hop = new_ids_2.nonzero(as_tuple=False).T
    edge_attr_two_hop = torch.ones(size=(edge_index_two_hop.size(-1), ),
                                   dtype=torch.long,
                                   device=bond_edge_index.device) * (BOND_FEATURE_DIMS + 2)
    
    adj3 = binarize(adj2 @ adj1) * no_self_loop
    
    new_ids_3 = (1.0 - adj1) * adj3
    new_ids_3 = (1.0 - adj2) * new_ids_3 
    
    edge_index_three_hop = new_ids_3.nonzero(as_tuple=False).T
    edge_attr_three_hop = torch.ones(size=(edge_index_three_hop.size(-1), ),
                                     dtype=torch.long,
                                     device=bond_edge_index.device) * (BOND_FEATURE_DIMS + 3)
    
    
    ext_edge_index = torch.concat([bond_edge_index,
                                   edge_index_two_hop,
                                   edge_index_three_hop], 
                                  dim=-1)
    ext_edge_attr = torch.concat([bond_edge_attr,
                                  edge_attr_two_hop,
                                  edge_attr_three_hop],
                                dim=-1)
    
    if check_coalesce:
        # make sure there are no "duplicates", i.e. length does not change after aggr.
        ext_edge_index_c, ext_edge_attr_c = coalesce(ext_edge_index,
                                                    ext_edge_attr,
                                                    m=len(batch),
                                                    n=len(batch))
        assert ext_edge_index.size() == ext_edge_index_c.size()
        assert ext_edge_attr.size() == ext_edge_attr_c.size()
    
    ext_edge_index, ext_edge_attr = sort_edge_index(edge_index=ext_edge_index,
                                                    edge_attr=ext_edge_attr,
                                                    num_nodes=len(batch),
                                                    sort_by_row=False)
    
    return ext_edge_index, ext_edge_attr


class Trainer(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        
        self.model = ScoreModelCoords(
            hn_dim=(hparams["sdim"], hparams["vdim"]),
            t_dim=hparams["tdim"],
            edge_dim=hparams["edim"],
            cutoff=hparams["cutoff"],
            rbf_dim=hparams["rbf_dim"],
            num_layers=hparams["num_layers"],
            use_norm=not hparams["omit_norm"],
            use_cross_product=not hparams["omit_cross_product"],
            use_all_atom_features=hparams["use_all_atom_features"],
            vector_aggr=hparams["vector_aggr"],
            fully_connected=hparams["fully_connected"],
            local_global_model=hparams["local_global_model"]
        )
        
        self.radius_graph = False

        timesteps = torch.arange(hparams["num_diffusion_timesteps"], dtype=torch.long)
        timesteps_embedder = get_timestep_embedding(
            timesteps=timesteps, embedding_dim=hparams["tdim"]
        ).to(torch.float32)

        self.register_buffer("timesteps", tensor=timesteps)
        self.register_buffer("timesteps_embedder", tensor=timesteps_embedder)
        
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
                                    scaled_reverse_posterior_sigma=True,
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
                          dtype=self.model.atom_encoder.atom_embedding_list[0].weight.dtype)
        pos = pos - scatter_mean(pos, dim=0, index=batch, dim_size=bs)[batch]
        
        if self.radius_graph:
            edge_index_local = radius_graph(x=pos,
                                            r=self.hparams.cutoff,
                                            batch=batch,
                                            max_num_neighbors=self.hparams.max_num_neighbors
                                            )
        else:
            edge_index_local, edge_attr_local = triple_order_edges(bond_edge_index=bond_edge_index,
                                                                   bond_edge_attr=bond_edge_attr,
                                                                   batch=batch, check_coalesce=False
                                                                   )
        
        
        edge_index_global = torch.eq(batch.unsqueeze(0), batch.unsqueeze(-1)).int().fill_diagonal_(0)
        edge_index_global, _ = dense_to_sparse(edge_index_global)

        if self.hparams.use_bond_features:
            if self.radius_graph:
                edge_index_local, edge_attr_local = self.coalesce_edges(edge_index=edge_index_local,
                                                                        bond_edge_index=bond_edge_index,
                                                                        bond_edge_attr=bond_edge_attr,
                                                                        n=pos.size(0))
                
            edge_index_global, edge_attr_global = self.coalesce_edges(edge_index=edge_index_global,
                                                                      bond_edge_index=bond_edge_index,
                                                                      bond_edge_attr=bond_edge_attr,
                                                                      n=pos.size(0))

        pos_sde_traj = []
        pos_mean_traj = []
        
        chain = range(num_diffusion_timesteps)
        if verbose:
            print(chain)
        iterator = tqdm(reversed(chain), total=num_diffusion_timesteps) if verbose else reversed(chain)
        for timestep in iterator:
            t = torch.full(size=(bs, ), fill_value=timestep, dtype=torch.long, device=pos.device)
            temb = self.timesteps_embedder[t][batch]
            out = self.model(
                x=x,
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

            if not self.hparams.fully_connected:
                edge_index_local = radius_graph(x=pos.detach(),
                                                r=self.hparams.cutoff,
                                                batch=batch, 
                                                max_num_neighbors=self.hparams.max_num_neighbors)
                if self.hparams.use_bond_features:
                    edge_index_local, edge_attr_local = self.coalesce_edges(edge_index=edge_index_local,
                                                                            bond_edge_index=bond_edge_index,
                                                                            bond_edge_attr=bond_edge_attr,
                                                                            n=pos.size(0))
            if save_traj:
                pos_sde_traj.append(pos.detach())
                pos_mean_traj.append(pos_mean.detach())

        return pos, [pos_sde_traj, pos_mean_traj]

    def forward(self, batch: Batch, t: Tensor):
        node_feat: Tensor = batch.x
        pos: Tensor = batch.pos
        data_batch: Tensor = batch.batch
        bond_edge_index = batch.edge_index
        bond_edge_attr = batch.edge_attr

        bs = int(data_batch.max()) + 1
        
        if self.hparams.continuous:
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
        
        if self.radius_graph:
            edge_index_local = radius_graph(x=pos_perturbed,
                                            r=self.hparams.cutoff,
                                            batch=data_batch,
                                            max_num_neighbors=self.hparams.max_num_neighbors
                                            )
        else:
            edge_index_local, edge_attr_local = triple_order_edges(bond_edge_index=bond_edge_index,
                                                                   bond_edge_attr=bond_edge_attr,
                                                                   batch=batch, check_coalesce=False
                                                                   )
        
        edge_index_global = batch.edge_index_fc
        
        if self.hparams.use_bond_features:
            if self.radius_graph:
                edge_index_local, edge_attr_local = self.coalesce_edges(edge_index=edge_index_local,
                                                                        bond_edge_index=bond_edge_index, bond_edge_attr=bond_edge_attr,
                                                                        n=pos.size(0))
            
            edge_index_global, edge_attr_global = self.coalesce_edges(edge_index=edge_index_global,
                                                                      bond_edge_index=bond_edge_index, bond_edge_attr=bond_edge_attr,
                                                                      n=pos.size(0))
            
        out = self.model(
            x=node_feat,
            t=timestep_embs,
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
        loss = torch.pow(out_dict["pred_noise"] - out_dict["true_noise"], 2).mean(-1)
        loss = scatter_mean(loss, index=batch.batch, dim=0, dim_size=batch_size)
        loss = torch.mean(loss, dim=0) 
        
        if stage == "val":
            sync_dist =  self.hparams.gpus > 1
        else:
            sync_dist = False
            
        self.log(
            f"{stage}/loss",
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
        max_num_conformers=hparams.max_num_conformers,
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
        max_time=hparams.max_time
    )

    pl.seed_everything(seed=0, workers=hparams.gpus > 1)

    trainer.fit(
        model=model,
        datamodule=datamodule,
        ckpt_path=hparams.load_ckpt if hparams.load_ckpt != "" else None,
    )
