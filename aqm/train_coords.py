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
from e3moldiffusion.molfeat import atom_type_config, get_bond_feature_dims
from e3moldiffusion.sde import VPAncestralSamplingPredictor, DiscreteDDPM
from e3moldiffusion.coords import ScoreModel

from torch import Tensor
from torch_geometric.data import Batch
from torch_geometric.typing import OptTensor
from torch_geometric.nn import radius_graph
from torch_sparse import coalesce
from torch_geometric.utils import dense_to_sparse, sort_edge_index
from torch_scatter import scatter_mean, scatter_add
from tqdm import tqdm

#New imported 
#from aqm.analyze_sampling import SamplingMetrics
from callbacks.ema import ExponentialMovingAverage
from config_file import get_dataset_info
#from aqm.info_data import AQMInfos
#from evaluation.diffusion_distribution import (
#    get_distributions
#)

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

BOND_FEATURE_DIMS = get_bond_feature_dims()[0]


class Trainer(pl.LightningModule):
    def __init__(self,
                 hparams,
                 dataset_info=None,
                 dataset_statistics=None,
                 train_smiles=None,
                 properties_norm=None,
                 nodes_dist=None,
                 prop_dist=None,
                 ):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.include_charges = False
        self.num_atom_features = self.hparams.num_atom_types + int(self.include_charges)
        self.num_bond_classes = 5
        
        self.i = 0
        self.properties_norm = properties_norm
        self.dataset_info = dataset_info
        self.train_smiles = train_smiles
        self.nodes_dist = nodes_dist
        self.prop_dist = prop_dist
        
        
    
        #self.val_sampling_metrics = SamplingMetrics(
        #    self.train_smiles, dataset_statistics, test=False
        #)
        
        self.node_scaling = 0.25
        
        self.model = ScoreModel(
            hn_dim=(hparams["sdim"], hparams["vdim"]),
            num_layers=hparams["num_layers"],
            use_norm=hparams["use_norm"],
            use_cross_product=hparams["use_cross_product"],
            num_atom_types=self.num_atom_features,
            num_bond_types=self.num_bond_classes,
            edge_dim=min(hparams["sdim"] // 4, 16),
            rbf_dim=hparams["num_rbf"],
            cutoff_local=hparams["cutoff_upper"],
            #cutoff_global=hparams["cutoff_upper_global"],
            vector_aggr="mean",
            local_global_model=True, #hparams["fully_connected_layer"],
            fully_connected=False #hparams["fully_connected"]
        )

        self.sde = DiscreteDDPM(beta_min=hparams["beta_min"],
                                beta_max=hparams["beta_max"],
                                N=hparams["num_diffusion_timesteps"],
                                scaled_reverse_posterior_sigma=False,
                                schedule="cosine")
            
        self.sampler = VPAncestralSamplingPredictor(sde=self.sde)
     
    def coalesce_edges(self, edge_index, bond_edge_index, bond_edge_attr, n):
        edge_attr = torch.full(size=(edge_index.size(-1), ), fill_value=0, device=edge_index.device, dtype=torch.long)
        edge_index = torch.cat([edge_index, bond_edge_index], dim=-1)
        edge_attr =  torch.cat([edge_attr, bond_edge_attr], dim=0)
        edge_index, edge_attr = coalesce(index=edge_index, value=edge_attr, m=n, n=n, op="max")
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
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, List]:
        
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
        edge_index_global = sort_edge_index(edge_index_global, sort_by_row=False)

        if self.hparams.use_bond_features:    
            edge_index_global, edge_attr_global = self.coalesce_edges(edge_index=edge_index_global,
                                                                      bond_edge_index=bond_edge_index,
                                                                      bond_edge_attr=bond_edge_attr,
                                                                      n=pos.size(0))

        pos_sde_traj = []
        pos_mean_traj = []
        
        xohe = F.one_hot(x, num_classes=self.num_atom_features).float()
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
        
        node_feat: Tensor = batch.z
        pos: Tensor = batch.pos
        charges: Tensor = batch.charges
        data_batch: Tensor = batch.batch
        batch_num_nodes = torch.bincount(data_batch)
        bond_edge_index = batch.bond_index
        bond_edge_attr = batch.bond_attr
        n = batch.num_nodes
        bs = int(data_batch.max()) + 1
        
        bond_edge_index, bond_edge_attr = sort_edge_index(edge_index=bond_edge_index,
                                                          edge_attr=bond_edge_attr,
                                                          sort_by_row=False)
        
        if not hasattr(batch, "fc_edge_index"):
            edge_index_global = torch.eq(batch.batch.unsqueeze(0), batch.batch.unsqueeze(-1)).int().fill_diagonal_(0)
            edge_index_global, _ = dense_to_sparse(edge_index_global)
            edge_index_global = sort_edge_index(edge_index_global, sort_by_row=False)
        else:
            edge_index_global = batch.fc_edge_index
        
        edge_index_global, edge_attr_global = self.coalesce_edges(edge_index=edge_index_global,
                                                                  bond_edge_index=bond_edge_index, 
                                                                  bond_edge_attr=bond_edge_attr,
                                                                  n=pos.size(0))
        
        edge_index_global, edge_attr_global = sort_edge_index(edge_index=edge_index_global,
                                                              edge_attr=edge_attr_global, 
                                                              sort_by_row=False)
      
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
        xohe = F.one_hot(
            node_feat.squeeze().long(), num_classes=max(self.hparams["atom_types"]) + 1
        ).float()[:, self.hparams["atom_types"]]
             
        edge_attr_local = F.one_hot(bond_edge_attr, num_classes=BOND_FEATURE_DIMS + 1).float()
        edge_attr_global = F.one_hot(edge_attr_global, num_classes=BOND_FEATURE_DIMS + 1).float()
       
        out = self.model(
            x=xohe,
            t=temb,
            pos=pos_perturbed,
            edge_index_local=bond_edge_index,
            edge_index_global=edge_index_global,
            edge_attr_local=edge_attr_local,
            edge_attr_global=edge_attr_global,
            batch=data_batch
        )

        noise_coords_pred = out["score_coords"]
        noise_coords_pred = zero_mean(noise_coords_pred, batch=data_batch, dim_size=bs, dim=0)
        
        
        out = {
            "noise_coords_pred": noise_coords_pred,
            "noise_coords_true": noise_coords_true
            }
        
        return out
    
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
        loss = torch.pow(out_dict["noise_coords_pred"] - out_dict["noise_coords_true"], 2).sum(-1)
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
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams["lr"])
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

    

if __name__ == "__main__":
    from aqm.data import AQMDataModule
    from aqm.hparams_coordsatomsbonds import add_arguments

    parser = ArgumentParser()
    parser = add_arguments(parser)
    hparams = parser.parse_args()
    
    if not os.path.exists(hparams.log_dir):
        os.makedirs(hparams.log_dir)

    if not os.path.isdir(hparams.log_dir + f"/run{hparams.id}/"):
        print("Creating directory")
        os.mkdir(hparams.log_dir + f"/run{hparams.id}/")
    print(f"Starting Run {hparams.id}")
    
    ema_callback = ExponentialMovingAverage(decay=hparams.ema_decay)
    checkpoint_callback = ModelCheckpoint(
        dirpath=hparams.log_dir + f"/run{hparams.id}/",
        save_top_k=1,
        monitor="val/coords_loss",
        save_last=True,
    )
    lr_logger = LearningRateMonitor()
    tb_logger = TensorBoardLogger(
        hparams.log_dir + f"/run{hparams.id}/", default_hp_metric=False
    )

    print(f"Loading {hparams.dataset} Datamodule.")
    datamodule = AQMDataModule(hparams)
    datamodule.prepare_data()
    datamodule.setup("fit")

    #dataset_statistics = AQMInfos(datamodule.dataset)
    dataset_statistics = None

    smiles = datamodule.dataset.data.smiles
    train_idx = [int(i) for i in datamodule.idx_train]
    train_smiles = [smi for i, smi in enumerate(smiles) if i in train_idx]
    dataset_info = get_dataset_info(hparams.dataset, hparams.remove_hs)

    properties_norm = None
    if len(hparams.properties_list) > 0:
        properties_norm = datamodule.compute_mean_mad(hparams.properties_list)

    dataloader = datamodule.get_dataloader(datamodule.train_dataset, "val")
    #nodes_dist, prop_dist = get_distributions(hparams, dataset_info, dataloader)
    nodes_dist, prop_dist = None, None
    
    if prop_dist is not None:
        prop_dist.set_normalizer(properties_norm)

    model = Trainer(
        hparams=hparams.__dict__,
        dataset_info=dataset_info,
        dataset_statistics=dataset_statistics,
        train_smiles=train_smiles,
        nodes_dist=nodes_dist,
        prop_dist=prop_dist,
        properties_norm=properties_norm,
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
            ema_callback,
            lr_logger,
            checkpoint_callback,
            TQDMProgressBar(refresh_rate=5),
            ModelSummary(max_depth=2),
        ],
        precision=hparams.precision,
        num_sanity_val_steps=2,
        max_epochs=hparams.num_epochs,
        detect_anomaly=hparams.detect_anomaly,
        resume_from_checkpoint=None
        if hparams.load_model is None
        else hparams.load_model,
    )

    pl.seed_everything(seed=0, workers=hparams.gpus > 1)

    trainer.fit(
        model=model,
        datamodule=datamodule,
        # ckpt_path=hparams.load_ckpt if hparams.load_ckpt != "" else None,
    )
