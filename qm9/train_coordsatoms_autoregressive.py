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
from e3moldiffusion.autoregressive import E3ARDiffusionModel

from torch import Tensor
from torch_geometric.data import Batch
from torch_geometric.typing import OptTensor
from torch_geometric.nn import radius_graph
from torch_sparse import coalesce
from torch_geometric.utils import dense_to_sparse, sort_edge_index
from torch_scatter import scatter_mean, scatter_add
from tqdm import tqdm

#New imported 
from callbacks.ema import ExponentialMovingAverage
from config_file import get_dataset_info
from qm9.info_data import QM9Infos

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

                 ):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.num_atom_features = self.hparams.num_atom_types
        self.i = 0
        self.dataset_info = dataset_info

        self.model = E3ARDiffusionModel(
            hn_dim=(hparams["sdim"], hparams["vdim"]),
            num_layers=hparams["num_layers"],
            use_norm=hparams["use_norm"],
            use_cross_product=hparams["use_cross_product"],
            num_atom_types=self.num_atom_features,
            vector_aggr="mean",
            edge_dim=None,
            mask=["coords", "atoms"]
        )

    def _get_empirical_num_nodes(self):
        num_nodes_dict = self.dataset_info.get('n_nodes')
        max_num_nodes = max(num_nodes_dict.keys())
        empirical_distribution_num_nodes = {i: num_nodes_dict.get(i) for i in range(max_num_nodes)}
        empirical_distribution_num_nodes_tensor = {}

        for key, value in empirical_distribution_num_nodes.items():
            if value is None:
                value = 0
            empirical_distribution_num_nodes_tensor[key] = value
        empirical_distribution_num_nodes_tensor = torch.tensor(list(empirical_distribution_num_nodes_tensor.values())).float()
        return empirical_distribution_num_nodes_tensor
                
    def get_list_of_edge_adjs(self, edge_attrs_dense, batch_num_nodes):
        ptr = torch.cat([torch.zeros(1, device=batch_num_nodes.device, dtype=torch.long), batch_num_nodes.cumsum(0)])
        edge_tensor_lists = []
        for i in range(len(ptr) - 1):
            select_slice = slice(ptr[i].item(), ptr[i+1].item())
            e = edge_attrs_dense[select_slice, select_slice]
            edge_tensor_lists.append(e)
        return edge_tensor_lists
    
    
    def reverse_sampling(
        self,
        num_graphs: int,
        num_nodes: int,
        device: torch.device,
        save_traj: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, List]:
        
        batch_num_nodes = torch.tensor([num_nodes for _ in range(num_graphs)]).long().to(device)
        batch = torch.arange(num_graphs, device=device).repeat_interleave(batch_num_nodes, dim=0)
        bs = int(batch.max()) + 1
        
        # initialiaze the 0-mean point cloud from N(0, I)
        pos = torch.randn(len(batch), 3,
                          device=device,
                          dtype=torch.get_default_dtype()
                          )
        pos = zero_mean(pos, batch=batch, dim_size=bs, dim=0)
        
        # initialize the atom-types 
        atom_types = torch.zeros(pos.size(0), self.num_atom_features, device=device)
        edge_index_global = torch.eq(batch.unsqueeze(0), batch.unsqueeze(-1)).int().fill_diagonal_(0)
        edge_index_global, _ = dense_to_sparse(edge_index_global)
        edge_index_global = sort_edge_index(edge_index_global, sort_by_row=False)

        perm = torch.cat([torch.randperm(num_nodes, device=device) for _ in range(bs)])
        t_placeholder = torch.empty(num_nodes, device=device, dtype=torch.long).fill_(0)
        pos_list = [pos]
        atom_list = [atom_types] 
        for t in range(num_nodes):
            t_placeholder = t_placeholder.fill_(t)
            mask = (perm < t_placeholder)
            sampled = (perm == t_placeholder).float().unsqueeze(-1)
            
            temb = (t_placeholder.float() / num_nodes).unsqueeze(-1)
            
            out = self.model(
            x=atom_types,
            t=temb,
            mask=mask,
            pos=pos,
            edge_index=edge_index_global,
            edge_attr=None,
            batch=batch
        )
            # update positions
            pos = (1.0 - sampled) * pos + (sampled) * out["coords_mean"]
            atom_types = (1.0 - sampled) * atom_types + (sampled) * out["atoms_pred"]
            atom_types = atom_types.argmax(dim=-1)
            if save_traj:
                pos_list.append(pos.detach())
                atom_list.append(atom_types)
            atom_types = F.one_hot(atom_types, num_classes=self.num_atom_features)
            
        return pos, (pos_list, atom_list)
            
            
            
    
    def forward(self, batch: Batch):
        
        node_feat: Tensor = batch.z
        pos: Tensor = batch.pos
        data_batch: Tensor = batch.batch
        batch_num_nodes = torch.bincount(data_batch)
        bs = int(data_batch.max()) + 1

        pos = zero_mean(pos, data_batch, dim_size=bs, dim=0)
        
        t = torch.cat([torch.randint(low=0, high=n, size=(1, )) for n in batch_num_nodes.cpu().tolist()])
        perm = torch.cat([torch.randperm(n,) for n in batch_num_nodes.tolist()])
        t, perm = t.to(node_feat.device), perm.to(node_feat.device)

        t_ = t[data_batch]
        
        mask = (perm < t_)
        
        temb = t_.float() / batch_num_nodes[data_batch].float()
        temb = temb.unsqueeze(dim=1)
        
        # one-hot-encode
        xohe = F.one_hot(
            node_feat.squeeze().long(), num_classes=max(self.hparams["atom_types"]) + 1
        ).float()[:, self.hparams["atom_types"]]
        
        if not hasattr(batch, "fc_edge_index"):
            edge_index_global = torch.eq(batch.batch.unsqueeze(0), batch.batch.unsqueeze(-1)).int().fill_diagonal_(0)
            edge_index_global, _ = dense_to_sparse(edge_index_global)
            edge_index_global = sort_edge_index(edge_index_global, sort_by_row=False)
        else:
            edge_index_global = batch.fc_edge_index

        out = self.model(
            x=xohe,
            t=temb,
            mask=mask,
            pos=pos,
            edge_index=edge_index_global,
            edge_attr=None,
            batch=data_batch
        )

        atoms_pred = out["atoms_pred"]
        coords_pred = out["coords_mean"]
        
        out = {
            "coords_pred": coords_pred,
            "coords_true": pos,
            "atoms_pred": atoms_pred,
            "atoms_true": xohe.argmax(-1),
            "mask": mask,
            "perm": perm,
            "t": t,
            "batch_num_nodes": batch_num_nodes
        }
        
        return out
    
    def step_fnc(self, batch, batch_idx, stage: str):
        batch_size = int(batch.batch.max()) + 1    
        out_dict = self(batch=batch)
        
        f = (1.0 - out_dict["mask"].float())
        s = out_dict["batch_num_nodes"].float() / (out_dict["batch_num_nodes"].float() - out_dict["t"] + 1)
        
        coords_loss = torch.pow(
            out_dict["coords_pred"] - out_dict["coords_true"], 2
        ).sum(-1)
        coords_loss *= f
        coords_loss = scatter_add(
            coords_loss, index=batch.batch, dim=0, dim_size=batch_size
        )
        coords_loss *= s
        coords_loss = torch.mean(coords_loss, dim=0)
        
        
        atoms_loss = F.cross_entropy(
            out_dict["atoms_pred"], out_dict["atoms_true"], reduction='none'
            )
        atoms_loss *= f
        atoms_loss = scatter_add(
            atoms_loss, index=batch.batch, dim=0, dim_size=batch_size
        )
        atoms_loss *= s
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
    from qm9.data import QM9DataModule
    from qm9.hparams_coordsatomsbonds import add_arguments

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
    datamodule = QM9DataModule(hparams)
    datamodule.prepare_data()
    datamodule.setup("fit")
    
    dataset_info = get_dataset_info(hparams.dataset, hparams.remove_hs)

    model = Trainer(
        hparams=hparams.__dict__,
        dataset_info=dataset_info,
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
