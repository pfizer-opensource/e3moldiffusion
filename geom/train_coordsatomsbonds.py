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
from e3moldiffusion.sde import VPSDE, VPAncestralSamplingPredictor, DiscreteDDPM
from e3moldiffusion.coordsatomsbonds import ScoreModel

from torch import Tensor
from torch_geometric.data import Batch
from torch_geometric.typing import OptTensor
from torch_geometric.nn import radius_graph
from torch_sparse import coalesce
from torch_geometric.utils import dense_to_sparse, sort_edge_index
from torch_scatter import scatter_mean, scatter_add
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

BOND_FEATURE_DIMS = get_bond_feature_dims()[0]


class Trainer(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.hparams.num_atom_types = get_num_atom_types_geom(dataset=hparams["dataset"])
        self.hparams.num_bond_types = BOND_FEATURE_DIMS + 1

        self.edge_scaling = 1.00
        self.node_scaling = 0.25
        
        self.model = ScoreModel(
            num_bond_types=BOND_FEATURE_DIMS + 1,
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
        empirical_distribution_num_nodes: Tensor,
        device: torch.device,
        verbose: bool = False,
        save_traj: bool = False
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
        # sample symmetric edge-attributes
        edge_attrs = torch.randn((edge_index_global.size(0),
                                  edge_index_global.size(1),
                                  self.hparams.num_bond_types),
                                  device=device, 
                                  dtype=torch.get_default_dtype())
        # symmetrize
        edge_attrs = 0.5 * (edge_attrs + edge_attrs.permute(1, 0, 2))
        assert torch.norm(edge_attrs - edge_attrs.permute(1, 0, 2)).item() == 0.0
        # get COO format (2, E)
        edge_index_global, _ = dense_to_sparse(edge_index_global)
        edge_index_global = sort_edge_index(edge_index_global, sort_by_row=False)
        # select in PyG formt (E, self.hparams.num_bond_types)
        edge_attr_global = edge_attrs[edge_index_global[0, :], edge_index_global[1, :], :]
        batch_edge = batch[edge_index_global[0]]     


        pos_traj = []
        atom_type_traj = []
        edge_type_traj = []
        
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
                edge_attr_global=edge_attr_global,
                batch=batch,
                batch_edge=batch_edge
            )
             
            score_coords = out["score_coords"]
            score_atoms = out["score_atoms"]
            score_bonds = out["score_bonds"]
            
            score_coords = zero_mean(score_coords, batch=batch, dim_size=bs, dim=0)
            noise_coords = torch.randn_like(pos)
            noise_coords = zero_mean(noise_coords, batch=batch, dim_size=bs, dim=0)

            # coords
            pos, _ = self.sampler.update_fn(x=pos, score=score_coords, t=t[batch], noise=noise_coords)
            pos = zero_mean(pos, batch=batch, dim_size=bs, dim=0)
            
            # atoms
            noise_atoms = torch.randn_like(atom_types)
            atom_types, _ = self.sampler.update_fn(x=atom_types, score=score_atoms, t=t[batch], noise=noise_atoms)
            
            # bonds
            noise_edges = torch.randn_like(edge_attrs)
            noise_edges = 0.5 * (noise_edges + noise_edges.permute(1, 0, 2))
            noise_edges = noise_edges[edge_index_global[0, :], edge_index_global[1, :], :]
            edge_attr_global, _ = self.sampler.update_fn(x=edge_attr_global, score=score_bonds,
                                                         t=t[batch_edge], noise=noise_edges)
            
            
            if not self.hparams.fully_connected:
                edge_index_local = radius_graph(x=pos.detach(),
                                                r=self.hparams.cutoff_local,
                                                batch=batch, 
                                                max_num_neighbors=self.hparams.max_num_neighbors)
            
            #atom_integer = torch.argmax(atom_types, dim=-1)
            #bond_integer = torch.argmax(edge_attr_global, dim=-1)
            
            if save_traj:
                pos_traj.append(pos.detach())
                atom_type_traj.append(atom_types.detach())
                edge_type_traj.append(edge_attr_global.detach())
                
        return pos, atom_types, edge_attr_global, batch_num_nodes, [pos_traj, atom_type_traj, edge_type_traj]
    
    def coalesce_edges(self, edge_index, bond_edge_index, bond_edge_attr, n):
        # possibly combine the bond-edge-index with radius graph or fully-connected graph
        # Note: This scenario is useful when learning the 3D coordinates only. 
        # From an optimization perspective, atoms that are connected by topology should have certain distance values. 
        # Since the atom types are fixed here, we know which molecule we want to generate a 3D configuration from, so the edge-index will help as inductive bias
        edge_attr = torch.full(size=(edge_index.size(-1), ), fill_value=0, device=edge_index.device, dtype=torch.long)
        # combine
        edge_index = torch.cat([edge_index, bond_edge_index], dim=-1)
        edge_attr =  torch.cat([edge_attr, bond_edge_attr], dim=0)
        # coalesce, i.e. reduce and remove duplicate entries by taking the maximum value, making sure that the bond-features are included. For No bonds we have included fill-value 0 anyways
        edge_index, edge_attr = coalesce(index=edge_index, value=edge_attr, m=n, n=n, op="max")
        return edge_index, edge_attr
    
    def forward(self, batch: Batch, t: Tensor):
        node_feat: Tensor = batch.xgeom
        pos: Tensor = batch.pos
        data_batch: Tensor = batch.batch
        bs = int(data_batch.max()) + 1
        n = batch.num_nodes
        batch_num_nodes = torch.bincount(data_batch)
        
        bond_edge_index = batch.edge_index
        bond_edge_attr = batch.edge_attr
        # increment by 1, because we want 0 to stand for no-edge
        bond_edge_attr_p1 = bond_edge_attr + 1
        if not hasattr(batch, "edge_index_fc"):
            edge_index_global = torch.eq(batch.batch.unsqueeze(0), batch.batch.unsqueeze(-1)).int().fill_diagonal_(0)
            edge_index_global, _ = dense_to_sparse(edge_index_global)
            edge_index_global = sort_edge_index(edge_index_global, sort_by_row=False)
        else:
            edge_index_global = batch.edge_index_fc
        
        edge_index_global, edge_attr_global = self.coalesce_edges(edge_index=edge_index_global,
                                                                  bond_edge_index=bond_edge_index, 
                                                                  bond_edge_attr=bond_edge_attr_p1,
                                                                  n=pos.size(0))
        
    
        # create block diagonal matrix
        dense_edge = torch.zeros(n, n, device=pos.device, dtype=torch.long)
        # populate entries with integer features 
        dense_edge[edge_index_global[0, :], edge_index_global[1, :]] = edge_attr_global        
        dense_edge_ohe = F.one_hot(dense_edge.view(-1, 1),
                                   num_classes=BOND_FEATURE_DIMS + 1).view(n, n, -1).float()
        
        # create symmetric noise for edge-attributes
        noise_edges = torch.randn_like(dense_edge_ohe)
        noise_edges = 0.5 * (noise_edges + noise_edges.permute(1, 0, 2))
        
        # PyG format
        batch_edge = data_batch[edge_index_global[0]]     
        noise_edge_attr = noise_edges[edge_index_global[0, :], edge_index_global[1, :], :]
        edge_attr_global = dense_edge_ohe[edge_index_global[0, :], edge_index_global[1, :], :]
        edge_attr_global = self.edge_scaling * edge_attr_global
        # Perturb
        mean_edges, std_edges = self.sde.marginal_prob(x=edge_attr_global, t=t[batch_edge])
        edge_attr_perturbed = mean_edges + std_edges * noise_edge_attr
        
        #signal = self.sde.sqrt_alphas_cumprod[t]
        #std = self.sde.sqrt_1m_alphas_cumprod[t]
        #signal = signal.repeat_interleave(batch_num_nodes).unsqueeze(-1)
        #std = std.repeat_interleave(batch_num_nodes).unsqueeze(-1)
        #dense_edge_ohe_perturbed = self.edge_scaling * dense_edge_ohe * signal + noise_edges * std
    
        # retrieve as edge-attributes in PyG Format 
        #edge_attr_perturbed = dense_edge_ohe_perturbed[edge_index_global[0, :], edge_index_global[1, :], :]
        #noise_edge_attr = noise_edges[edge_index_global[0, :], edge_index_global[1, :], :]
        #batch_edge = data_batch[edge_index_global[0]]     
        
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
        xohe = F.one_hot(node_feat, num_classes=self.hparams.num_atom_types).float()
        xohe = self.node_scaling * xohe
        # sample noise for OHEs in {0, 1}^NUM_CLASSES
        noise_ohes_true = torch.randn_like(xohe)
        mean_ohes, std_ohes = self.sde.marginal_prob(x=xohe, t=t[data_batch])
        # perturb OHEs
        ohes_perturbed = mean_ohes + std_ohes * noise_ohes_true
        
        edge_index_local = radius_graph(x=pos_perturbed,
                                        r=self.hparams.cutoff_local,
                                        batch=data_batch, 
                                        flow="source_to_target",
                                        max_num_neighbors=self.hparams.max_num_neighbors)
        
        # NEW: local edge-attributes
        # check where edge_index_local is in edge_index_global
        
        tryout = (edge_index_local.unsqueeze(-1) == edge_index_global.unsqueeze(1))
        local_global_idx_select = (tryout.sum(0) == 2).nonzero()[:, 1]
        # create mask tensor for backpropagating only local edges
        local_edge_mask = torch.zeros(size=(edge_index_global.size(1), ), dtype=torch.bool)
        local_edge_mask[local_global_idx_select] = True
        assert len(local_global_idx_select) == sum(local_edge_mask)
        edge_attr_local = edge_attr_global[edge_index_local, :]
        
        # check: to dense
        
        
        
        out = self.model(
            x=ohes_perturbed,
            t=temb,
            pos=pos_perturbed,
            edge_index_local=edge_index_local,
            edge_index_global=edge_index_global,
            edge_attr_local=edge_attr_local,
            edge_attr_global=edge_attr_perturbed,
            batch=data_batch,
            batch_edge=batch_edge
        )

        noise_ohes_atoms = out["score_atoms"]
        noise_coords_pred = out["score_coords"]
        noise_coords_pred = zero_mean(noise_coords_pred, batch=data_batch, dim_size=bs, dim=0)
        
        noise_ohes_bonds = out["score_bonds"]
        
        out = {
            "noise_coords_pred": noise_coords_pred,
            "noise_coords_true": noise_coords_true,
            "noise_atoms_pred": noise_ohes_atoms,
            "noise_atoms_true": noise_ohes_true,
            "true_atom_class": node_feat,
            "noise_bonds_pred": noise_ohes_bonds,
            "noise_bonds_true": noise_edge_attr,
            "true_edge_attr": edge_attr_global,
            "edge2batch": batch_edge
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
        
        bonds_loss = torch.pow(
            out_dict["noise_bonds_pred"] - out_dict["noise_bonds_true"], 2
        ).mean(-1) 
        
        #bonds_loss = bonds_loss.mean()
        bonds_loss = 0.5 * scatter_add(
            bonds_loss, index=out_dict["edge2batch"], dim=0, dim_size=out_dict["noise_atoms_pred"].size(0)
        )
        bonds_loss = scatter_mean(
            bonds_loss, index=batch.batch, dim=0, dim_size=batch_size
        )
        bonds_loss = bonds_loss.mean(dim=0)
        
        loss = coords_loss + atoms_loss + bonds_loss

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
        
        self.log(
            f"{stage}/bonds_loss",
            bonds_loss,
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
