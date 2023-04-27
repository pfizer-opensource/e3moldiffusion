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
from e3moldiffusion.coordsatomsbonds import ScoreModel

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
        
        self.edge_scaling = 1.00
        self.node_scaling = 0.25
        
        self.model = ScoreModel(
            hn_dim=(hparams["sdim"], hparams["vdim"]),
            num_layers=hparams["num_layers"],
            use_norm=hparams["use_norm"],
            use_cross_product=hparams["use_cross_product"],
            num_atom_types=self.num_atom_features,
            num_bond_types=self.num_bond_classes,
            rbf_dim=hparams["num_rbf"],
            cutoff_local=hparams["cutoff_upper"],
            cutoff_global=hparams["cutoff_upper_global"],
            vector_aggr="mean",
            local_global_model=hparams["fully_connected_layer"],
            fully_connected=hparams["fully_connected"],
            local_edge_attrs=hparams["use_local_edge_attr"]
        )

        self.sde = DiscreteDDPM(beta_min=hparams["beta_min"],
                                beta_max=hparams["beta_max"],
                                N=hparams["num_diffusion_timesteps"],
                                scaled_reverse_posterior_sigma=False,
                                schedule="cosine")
            
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
                                        r=self.hparams.cutoff_upper,
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
        
        # include local (noisy) edge-attributes based on radius graph indices
        ## old: the below takes a lot of gpu memory
        create_mask = False
        if create_mask:
            local_global_idx_select = (edge_index_local.unsqueeze(-1) == edge_index_global.unsqueeze(1))
            local_global_idx_select = (local_global_idx_select.sum(0) == 2).nonzero()[:, 1]
            edge_attr_local = edge_attr_global[edge_index_local, :]
        else:
            edge_attr_local = edge_attrs[edge_index_local[0], edge_index_local[1], :]


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
                edge_attr_local=edge_attr_local,
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
                                                r=self.hparams.cutoff_upper,
                                                batch=batch, 
                                                max_num_neighbors=self.hparams.max_num_neighbors)
                
                 # include local (noisy) edge-attributes based on radius graph indices
                if create_mask:
                    local_global_idx_select = (edge_index_local.unsqueeze(-1) == edge_index_global.unsqueeze(1))
                    local_global_idx_select = (local_global_idx_select.sum(0) == 2).nonzero()[:, 1]
                    edge_attr_local = edge_attr_global[edge_index_local, :]
                else:
                    # need to populate the dense edge-attr tensor
                    edge_attrs = torch.zeros_like(edge_attrs)
                    edge_attrs[edge_index_global[0], edge_index_global[1], :] = edge_attr_global
                    edge_attr_local = edge_attrs[edge_index_local[0], edge_index_local[1], :]
                    
            #atom_integer = torch.argmax(atom_types, dim=-1)
            #bond_integer = torch.argmax(edge_attr_global, dim=-1)
            
            if save_traj:
                pos_traj.append(pos.detach())
                atom_type_traj.append(atom_types.detach())
                edge_type_traj.append(edge_attr_global.detach())
                
        return pos, atom_types, edge_attr_global, batch_num_nodes, [pos_traj, atom_type_traj, edge_type_traj]
    
    def coalesce_edges(self, edge_index, bond_edge_index, bond_edge_attr, n):
        edge_attr = torch.full(size=(edge_index.size(-1), ),
                               fill_value=0,
                               device=edge_index.device,
                               dtype=torch.long)
        edge_index = torch.cat([edge_index, bond_edge_index], dim=-1)
        edge_attr =  torch.cat([edge_attr, bond_edge_attr], dim=0)
        edge_index, edge_attr = coalesce(index=edge_index, value=edge_attr, m=n, n=n, op="max")
        return edge_index, edge_attr
    
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

        # create block diagonal matrix
        dense_edge = torch.zeros(n, n, device=pos.device, dtype=torch.long)
        # populate entries with integer features 
        dense_edge[edge_index_global[0, :], edge_index_global[1, :]] = edge_attr_global        
        dense_edge_ohe = F.one_hot(dense_edge.view(-1, 1),
                                   num_classes=BOND_FEATURE_DIMS + 1).view(n, n, -1).float()
        
        assert torch.norm(dense_edge_ohe - dense_edge_ohe.permute(1, 0, 2)).item() == 0.0
        # edge-scaling
        dense_edge_ohe = self.edge_scaling * dense_edge_ohe
        
        # create symmetric noise for edge-attributes
        noise_edges = torch.randn_like(dense_edge_ohe)
        noise_edges = 0.5 * (noise_edges + noise_edges.permute(1, 0, 2))
        assert torch.norm(noise_edges - noise_edges.permute(1, 0, 2)).item() == 0.0
        
        # PyG format
        #batch_edge = data_batch[edge_index_global[0]]     
        #noise_edge_attr = noise_edges[edge_index_global[0, :], edge_index_global[1, :], :]
        #edge_attr_global = dense_edge_ohe[edge_index_global[0, :], edge_index_global[1, :], :]
        # Perturb
        #mean_edges, std_edges = self.sde.marginal_prob(x=edge_attr_global, t=t[batch_edge])
        #edge_attr_global_perturbed = mean_edges + std_edges * noise_edge_attr
        
        signal = self.sde.sqrt_alphas_cumprod[t]
        std = self.sde.sqrt_1m_alphas_cumprod[t]
        #signal = signal.repeat_interleave(batch_num_nodes).unsqueeze(-1)
        #std = std.repeat_interleave(batch_num_nodes).unsqueeze(-1)
        signal_b = signal[data_batch].unsqueeze(-1).unsqueeze(-1)
        std_b = std[data_batch].unsqueeze(-1).unsqueeze(-1)
        dense_edge_ohe_perturbed = dense_edge_ohe * signal_b + noise_edges * std_b
    
        # retrieve as edge-attributes in PyG Format 
        edge_attr_global_perturbed = dense_edge_ohe_perturbed[edge_index_global[0, :], edge_index_global[1, :], :]
        edge_attr_global_noise = noise_edges[edge_index_global[0, :], edge_index_global[1, :], :]
    
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
        
        xohe = self.node_scaling * xohe
        # sample noise for OHEs in {0, 1}^NUM_CLASSES
        noise_ohes_true = torch.randn_like(xohe)
        mean_ohes, std_ohes = self.sde.marginal_prob(x=xohe, t=t[data_batch])
        # perturb OHEs
        ohes_perturbed = mean_ohes + std_ohes * noise_ohes_true
        
        edge_index_local = radius_graph(x=pos_perturbed,
                                        r=self.hparams.cutoff_upper,
                                        batch=data_batch, 
                                        flow="source_to_target",
                                        max_num_neighbors=self.hparams.max_num_neighbors)
        
        create_mask = False
        if create_mask:
            local_global_idx_select = (edge_index_local.unsqueeze(-1) == edge_index_global.unsqueeze(1))
            local_global_idx_select = (local_global_idx_select.sum(0) == 2).nonzero()[:, 1]
            # create mask tensor for backpropagating only local edges
            local_edge_mask = torch.zeros(size=(edge_index_global.size(1), ), dtype=torch.bool)
            local_edge_mask[local_global_idx_select] = True
            assert len(local_global_idx_select) == sum(local_edge_mask)
            edge_attr_local_perturbed = edge_attr_global_perturbed[edge_index_local, :]
            # check: to dense: here just take the clean for checking.
            #edge_attr_local = edge_attr_global[local_global_idx_select, :]
            #edge_attr_local_ = dense_edge_ohe[edge_index_local[0, :], edge_index_local[1, :], :]
            #assert torch.norm(edge_attr_local - edge_attr_local_).item() == 0.0
        else:
            edge_attr_local_perturbed = dense_edge_ohe_perturbed[edge_index_local[0, :], edge_index_local[1, :], :]
            edge_attr_local_noise = noise_edges[edge_index_local[0, :], edge_index_local[1, :], :]
        
        batch_edge_global = data_batch[edge_index_global[0]]     
        batch_edge_local = data_batch[edge_index_local[0]]     
        
        out = self.model(
            x=ohes_perturbed,
            t=temb,
            pos=pos_perturbed,
            edge_index_local=edge_index_local,
            edge_index_global=edge_index_global,
            edge_attr_local=edge_attr_local_perturbed,
            edge_attr_global=edge_attr_global_perturbed,
            batch=data_batch,
            batch_edge_global=batch_edge_global,
            batch_edge_local=batch_edge_local,
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
            "noise_bonds_true": (edge_attr_local_noise, edge_attr_global_noise),
            "true_edge_attr": edge_attr_global,
            "edge_index": (edge_index_local, edge_index_global)
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
            out_dict["noise_bonds_pred"] - out_dict["noise_bonds_true"][1], 2
        ).mean(-1) 
        
        bonds_loss = 0.5 * scatter_mean(
            bonds_loss, index=out_dict["edge_index"][1][1], dim=0, dim_size=out_dict["noise_atoms_pred"].size(0)
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
