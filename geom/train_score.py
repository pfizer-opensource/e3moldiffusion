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
from e3moldiffusion.gsm import NonConservativeScoreNetwork, exact_jacobian_trace

from torch import Tensor
from torch_geometric.data import Batch
from torch_geometric.typing import OptTensor
from torch_geometric.utils import dense_to_sparse
from torch_sparse import coalesce
from torch_scatter import scatter_mean

logging.getLogger("lightning").setLevel(logging.WARNING)

BOND_FEATURE_DIMS = get_bond_feature_dims()[0]

class Trainer(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        
        self.model = NonConservativeScoreNetwork(
            hn_dim=(hparams["sdim"], hparams["vdim"]),
            edge_dim=hparams["edim"],
            cutoff=hparams["cutoff"],
            rbf_dim=hparams["rbf_dim"],
            num_layers=hparams["num_layers"],
            use_norm=not hparams["omit_norm"],
            use_cross_product=not hparams["omit_cross_product"],
            use_all_atom_features=hparams["use_all_atom_features"],
            vector_aggr=hparams["vector_aggr"],
            fully_connected=hparams["fully_connected"],
            local_global_model=hparams["local_global_model"],
            dist_score=hparams["dist_score"]
        )
 
    def coalesce_edges(self, edge_index, bond_edge_index, bond_edge_attr, n):
        edge_attr = torch.full(size=(edge_index.size(-1), ), fill_value=BOND_FEATURE_DIMS + 1, device=edge_index.device, dtype=torch.long)
        edge_index = torch.cat([edge_index, bond_edge_index], dim=-1)
        edge_attr =  torch.cat([edge_attr, bond_edge_attr], dim=0)
        edge_index, edge_attr = coalesce(index=edge_index, value=edge_attr, m=n, n=n, op="min")
        return edge_index, edge_attr
        
    def forward(self, batch: Batch):
        node_feat: Tensor = batch.x
        pos: Tensor = batch.pos
        data_batch: Tensor = batch.batch
        bond_edge_index = batch.edge_index
        bond_edge_attr = batch.edge_attr

        bs = int(data_batch.max()) + 1
        
        pos = pos - scatter_mean(pos, index=data_batch, dim=0, dim_size=bs)[data_batch]
        pos.requires_grad = True
        
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
            
        score = self.model(
            x=node_feat,
            pos=pos,
            edge_index_local=edge_index_local,
            edge_index_global=edge_index_global,
            edge_attr_local=edge_attr_local if self.hparams.use_bond_features else None,
            edge_attr_global=edge_attr_global if self.hparams.use_bond_features else None,
            batch=data_batch,
        )

        out_dict = {"sx": score, "x": pos}
        return out_dict
    
    def step_fnc(self, batch, batch_idx, stage):         
        batch_size = int(batch.batch.max()) + 1
       
        out_dict = self(batch=batch)
        score = out_dict["sx"]
        x = out_dict["x"]
        jcb_tr = exact_jacobian_trace(fx=score, x=x)
        
        snorm = torch.pow(score, 2).sum(-1).mean() 
        jcb_tr = jcb_tr.mean()
        loss = snorm + 2.0 * jcb_tr
        
        if stage == "val":
            sync_dist =  self.hparams.gpus > 1
        else:
            sync_dist = False
        
        self.log(
            f"{stage}/snorm",
            snorm,
            on_step=True,
            batch_size=batch_size,
            sync_dist=sync_dist
        )
        
        self.log(
            f"{stage}/jctr",
            jcb_tr,
            on_step=True,
            batch_size=batch_size,
            sync_dist=sync_dist
        )
        
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
        torch.set_grad_enabled(True)
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
