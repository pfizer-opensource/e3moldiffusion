import logging
import os
from argparse import ArgumentParser
from typing import List, Optional, Tuple
import json

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
        self.hparams.num_atom_types = get_num_atom_types_geom(dataset="drugs")
        self.num_atom_features = self.hparams.num_atom_types
        self.i = 0
        self.dataset_info = dataset_info

        self.model = E3ARDiffusionModel(
            hn_dim=(hparams["sdim"], hparams["vdim"]),
            num_layers=hparams["num_layers"],
            use_norm=not hparams["omit_norm"],
            use_cross_product=not hparams["omit_cross_product"],
            num_atom_types=self.num_atom_features,
            vector_aggr="mean"
        )
        
        empirical_num_nodes = self._get_empirical_num_nodes()
        self.register_buffer(name='empirical_num_nodes', tensor=empirical_num_nodes)
        
        
    def _get_empirical_num_nodes(self):
        if not self.hparams.no_h:
            with open('/home/let55/workspace/projects/e3moldiffusion/geom/num_nodes_geom_midi.json', 'r') as f:
                num_nodes_dict = json.load(f, object_hook=lambda d: {int(k) if k.lstrip('-').isdigit() else k: v for k, v in d.items()})
        else:
            with open('/home/let55/workspace/projects/e3moldiffusion/geom/num_nodes_geom_midi_no_h.json', 'r') as f:
                num_nodes_dict = json.load(f, object_hook=lambda d: {int(k) if k.lstrip('-').isdigit() else k: v for k, v in d.items()})
                
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
        atom_types = torch.randn(pos.size(0), self.num_atom_features, device=device)
        
        edge_index_local = radius_graph(x=pos,
                                        r=self.hparams.cutoff_upper,
                                        batch=batch, 
                                        max_num_neighbors=self.hparams.max_num_neighbors)
        
        edge_index_global = torch.eq(batch.unsqueeze(0), batch.unsqueeze(-1)).int().fill_diagonal_(0)
        edge_index_global, _ = dense_to_sparse(edge_index_global)
        edge_index_global = sort_edge_index(edge_index_global, sort_by_row=False)

        pos_traj = []
        atom_type_traj = []
        atom_type_ohe_traj = []
        chain = range(self.hparams.timesteps)
    
        if verbose:
            print(chain)
        iterator = tqdm(reversed(chain), total=len(chain)) if verbose else reversed(chain)
        for timestep in iterator:
            t = torch.full(size=(bs, ), fill_value=timestep, dtype=torch.long, device=pos.device)
            temb = t / self.hparams.timesteps
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
                                                r=self.hparams.cutoff_upper,
                                                batch=batch, 
                                                max_num_neighbors=self.hparams.max_num_neighbors)
            
            atom_integer = torch.argmax(atom_types, dim=-1)
            
            if save_traj:
                pos_traj.append(pos.detach())
                atom_type_traj.append(atom_types.detach())
                atom_type_ohe_traj.append(atom_integer)
                
        return pos, atom_types, atom_integer, batch_num_nodes, [pos_traj, atom_type_traj, atom_type_ohe_traj]
    
    def forward(self, batch: Batch):
        
        node_feat: Tensor = batch.x
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
        xohe = F.one_hot(node_feat, num_classes=self.num_atom_features).float()
        
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
    from geom.hparams_coordsatomsbonds import add_arguments

    parser = ArgumentParser()
    parser = add_arguments(parser)
    hparams = parser.parse_args()
    
    if not os.path.exists(hparams.save_dir):
        os.makedirs(hparams.save_dir)

    if not os.path.isdir(hparams.save_dir + f"/run{hparams.id}/"):
        print("Creating directory")
        os.mkdir(hparams.save_dir + f"/run{hparams.id}/")
    print(f"Starting Run {hparams.id}")
    ema_callback = ExponentialMovingAverage(decay=hparams.ema_decay)
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

    dataset_info = get_dataset_info("drugs", remove_h=False)
  
    print(f"Loading {hparams.dataset} Datamodule.")
    old = False
    
    if old:
        from geom.data import GeomDataModule
        print("Using native GEOM")
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
    else:
        from geom.geom_dataset import GeomDataModule
        print("Using MIDI GEOM")
        if hparams.no_h:
            root = '/home/let55/workspace/projects/e3moldiffusion/geom/data_noH' 
        else:
            root = '/home/let55/workspace/projects/e3moldiffusion/geom/data'
        print(root)
        datamodule = GeomDataModule(root=root,
                                    batch_size=hparams.batch_size,
                                    num_workers=hparams.num_workers,
                                    pin_memory=True,
                                    persistent_workers=True,
                                    with_hydrogen=not hparams.no_h
                                    )
    
    dataset_info = get_dataset_info(hparams.dataset, False)

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
        resume_from_checkpoint=hparams.load_ckpt if hparams.load_ckpt != "" else None,
    )

    pl.seed_everything(seed=0, workers=hparams.gpus > 1)

    trainer.fit(
        model=model,
        datamodule=datamodule,
        ckpt_path=hparams.load_ckpt if hparams.load_ckpt != "" else None,
    )
