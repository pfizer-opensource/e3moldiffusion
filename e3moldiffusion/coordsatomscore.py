import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from tqdm import tqdm
from typing import List, Tuple

from e3moldiffusion.gnn import EncoderGNN
from e3moldiffusion.modules import DenseLayer, GatedEquivBlock
from e3moldiffusion.molfeat import get_atom_feature_dims
from e3moldiffusion.sde import (
    DiscreteDDPM,
    VPAncestralSamplingPredictor,
    get_timestep_embedding,
)
from torch import Tensor, nn
from torch_scatter import scatter_mean
from torch_geometric.nn.inits import reset

MAX_NUM_ATOM_FEATURES = get_atom_feature_dims()[0] + 1


def zero_mean(x: Tensor, batch: Tensor, dim_size: int, dim=0):
    out = x - scatter_mean(x, index=batch, dim=dim, dim_size=dim_size)[batch]
    return out


def assert_zero_mean(x: Tensor, batch: Tensor, dim_size: int, dim=0, eps: float = 1e-6):
    out = scatter_mean(x, index=batch, dim=dim, dim_size=dim_size).mean()
    return abs(out) < eps


class AtomEmbedding(nn.Module):
    def __init__(self, out_dim: int) -> None:
        super().__init__()

        self.out_dim = out_dim
        embedding = torch.randn(MAX_NUM_ATOM_FEATURES, out_dim)
        embedding /= embedding.norm(p=2, dim=-1, keepdim=True)
        self.embedding = nn.parameter.Parameter(embedding)

        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.data = torch.randn(MAX_NUM_ATOM_FEATURES, self.out_dim)
        self.embedding.data /= self.embedding.data.norm(p=2, dim=-1, keepdim=True)

    def forward(self, x: Tensor, norm_gradient: bool = True) -> Tensor:
        out = self.embedding[x]
        out_norm = out.norm(p=2, dim=-1, keepdim=True)
        if not norm_gradient:
            out_norm = out_norm.detach()
        out = out / out_norm
        return out


default_hparams: dict = {
    "sdim": 128,
    "tdim": 64,
    "rbf_dim": 32,
    "vdim": 16,
    "edim": 0,
    "cutoff": 10.0,
    "num_layers": 5,
    "omit_norm": False,
    "omit_cross_product": False,
    "vector_aggr": "mean",
    "beta_min": 1e-4,
    "beta_max": 2e-2,
    "num_diffusion_timesteps": 300,
    "schedule": "cosine",
    "patience": 5,
    "cooldown": 5,
    "factor": 0.75,
}

class CoordsAtomScoreTrainer(pl.LightningModule):
    def __init__(self, hparams: dict = default_hparams) -> None:
        super(CoordsAtomScoreTrainer, self).__init__()

        self.save_hyperparameters(hparams)
        
        self.atom_time_mapping = nn.Sequential(
            DenseLayer(
                self.hparams.tdim + MAX_NUM_ATOM_FEATURES,
                self.hparams.sdim,
                activation=nn.SiLU(),
            ),
            DenseLayer(self.hparams.sdim, self.hparams.sdim),
        )

        self.gnn = EncoderGNN(
            hn_dim=(self.hparams.sdim, self.hparams.vdim),
            cutoff=self.hparams.cutoff,
            rbf_dim=self.hparams.rbf_dim,
            edge_dim=0,
            num_layers=self.hparams.num_layers,
            use_norm=not self.hparams.omit_norm,
            use_cross_product=not self.hparams.omit_cross_product,
            vector_aggr=self.hparams.vector_aggr,
        )

        self.atom_types_lin = DenseLayer(self.hparams.sdim, MAX_NUM_ATOM_FEATURES)
        self.coords_lin = DenseLayer(self.hparams.vdim, 1, bias=False)

        timesteps = torch.arange(self.hparams.num_diffusion_timesteps, dtype=torch.long)
        timesteps_embedder = get_timestep_embedding(
            timesteps=timesteps, embedding_dim=self.hparams.tdim
        ).to(torch.float32)

        self.register_buffer("timesteps", tensor=timesteps)
        self.register_buffer("timesteps_embedder", tensor=timesteps_embedder)

        self.sde = DiscreteDDPM(
            beta_min=self.hparams.beta_min,
            beta_max=self.hparams.beta_max,
            N=self.hparams.num_diffusion_timesteps,
            scaled_reverse_posterior_sigma=True,
            schedule=self.hparams.schedule,
        )

        self.sampler = VPAncestralSamplingPredictor(sde=self.sde)

        self.ce_loss = nn.CrossEntropyLoss(reduction="none")

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.atom_time_mapping)
        self.gnn.reset_parameters()
        self.atom_types_lin.reset_parameters()
        self.coords_lin.reset_parameters()

    def reverse_sampling(
        self,
        num_graphs: int,
        empirical_distribution_atom_types: Tensor,
        empirical_distribution_num_nodes: Tensor,
        verbose: bool = False,
        save_traj: bool = False
    ) -> Tuple[Tensor, List]:
        
        device = self.timesteps.data.device
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
        
        # sample atom types from empirical distribution
        atom_types = torch.multinomial(input=empirical_distribution_num_nodes,
                                       num_samples=len(batch),
                                       replacement=True).to(device)
        
        # fully-connected graphs
        ptr = torch.cumsum(batch_num_nodes, dim=0)
        ptr = torch.concat([torch.zeros(1, device=ptr.device, dtype=ptr.dtype), ptr[:-1]], dim=0)       
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
        edge_index = torch.concat(edge_index_list, dim=-1).to(x.device)

        pos_traj = []
        atom_type_traj = []
        chain = range(self.hparams.num_diffusion_timesteps)
    
        if verbose:
            print(chain)
        iterator = tqdm(reversed(chain), total=len(chain)) if verbose else reversed(chain)
        for timestep in iterator:
            t = torch.full(size=(bs, ), fill_value=timestep, dtype=torch.long, device=pos.device)
            temb = self.timesteps_embedder[t][batch]
            pass
        return pos, [None, None]

    def forward(
        self, x: Tensor, pos: Tensor, t: Tensor, edge_index: Tensor, batch: Tensor
    ):      
        
        # only select atom-types
        x = x[:, 0]
        
        # one-hot-encode
        xohe = F.one_hot(x, num_classes=MAX_NUM_ATOM_FEATURES).float()
        
        source, target = edge_index
        batch_size = int(batch.max()) + 1
        temb = self.timesteps_embedder[t][batch]

        # Coords: point cloud in R^3
        # sample noise for coords and recenter
        noise_coords_true = torch.randn_like(pos)
        noise_coords_true = zero_mean(noise_coords_true, batch=batch, dim_size=batch_size, dim=0)
        # center the true point cloud
        pos_centered = zero_mean(pos, batch, dim=0, dim_size=batch_size)
        # get signal and noise coefficients for coords
        mean_coords, std_coords = self.sde.marginal_prob(x=pos_centered, t=t[batch])
        # perturb coords
        pos_perturbed = mean_coords + std_coords * noise_coords_true

        # sample noise for OHEs in {0, 1}^NUM_CLASSES
        noise_ohes_true = torch.randn_like(xohe)
        mean_ohes, std_ohes = self.sde.marginal_prob(x=xohe, t=t[batch])
        # perturb OHEs
        ohes_perturbed = mean_ohes + std_ohes * noise_ohes_true

        r = pos_perturbed[target] - pos_perturbed[source]
        d = torch.pow(r, 2).sum(-1).sqrt().clamp(min=1e-6)
        r_norm = torch.div(r, d.unsqueeze(-1))
        edge_attr = (d, r_norm, None)

        s_merged = torch.concat([ohes_perturbed, temb], dim=-1)
        
        # gnn input
        s = self.atom_time_mapping(s_merged)
        v = torch.zeros(size=(x.size(0), 3, self.hparams.vdim), device=s.device)

        out = self.gnn(
            s=s, v=v, edge_index=edge_index, edge_attr=edge_attr, batch=batch
        )
        
        noise_coords_pred = self.coords_lin(out["v"]).squeeze()
        noise_ohes_pred = self.atom_types_lin(out["s"])
         
        noise_coords_pred = zero_mean(noise_coords_pred, batch=batch, dim_size=batch_size, dim=0)

        out = {
            "noise_coords_pred": noise_coords_pred,
            "noise_coords_true": noise_coords_true,
            "noise_ohes_pred": noise_ohes_pred,
            "noise_ohes_true": noise_ohes_true,
            "true_class": x,
        }

        return out

    def step_fnc(self, batch, batch_idx, stage: str):
        batch_size = int(batch.batch.max()) + 1

        t = torch.randint(
            low=0,
            high=self.hparams.num_diffusion_timesteps,
            size=(batch_size,),
            dtype=torch.long,
            device=batch.x.device,
        )

        out_dict = self(x=batch.x, pos=batch.pos, t=t, edge_index=batch.edge_index_fc, batch=batch.batch)
        coords_loss = torch.pow(
            out_dict["noise_coords_pred"] - out_dict["noise_coords_true"], 2
        ).mean(-1)
        coords_loss = scatter_mean(
            coords_loss, index=batch.batch, dim=0, dim_size=batch_size
        )
        coords_loss = torch.mean(coords_loss, dim=0)
        
        ohes_loss = torch.pow(
            out_dict["noise_ohes_pred"] - out_dict["noise_ohes_true"], 2
        ).mean(-1) 
        ohes_loss = scatter_mean(
            ohes_loss, index=batch.batch, dim=0, dim_size=batch_size
        )
        ohes_loss = torch.mean(ohes_loss, dim=0)

        loss = 1/2 * (coords_loss + ohes_loss)

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
            f"{stage}/ohes_loss",
            ohes_loss,
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
    
    default_hparams: dict = {
        "sdim": 128,
        "tdim": 64,
        "rbf_dim": 32,
        "cutoff": 10.0,
        "edim": 0,
        "vdim": 16,
        "num_layers": 5,
        "omit_norm": False,
        "omit_cross_product": False,
        "vector_aggr": "mean",
        "beta_min": 1e-4,
        "beta_max": 2e-2,
        "num_diffusion_timesteps": 300,
        "schedule": "cosine",
        "patience": 5,
        "cooldown": 5,
        "factor": 0.75,
    }

    trainer = CoordsAtomScoreTrainer(hparams=default_hparams)

    print(sum(m.numel() for m in trainer.parameters() if m.requires_grad_))
