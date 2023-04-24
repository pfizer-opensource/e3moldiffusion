import copy
import math
import os
from contextlib import suppress
from pathlib import Path
from typing import Optional, Type, Union
import logging
import os
from argparse import ArgumentParser
from typing import List, Optional, Tuple
import torch.nn.functional as F
from callbacks.ema import ExponentialMovingAverage
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np
from torch import Tensor
from torch_geometric.data import Batch
from torch_geometric.typing import OptTensor
from torch_geometric.utils import to_dense_adj, to_dense_batch, remove_self_loops

from torch.nn.functional import mse_loss
from torch_cluster import radius_graph
from torch_sparse import coalesce
from torch_scatter import scatter_mean, scatter_add
from evaluation.diffusion_utils import (
    PredefinedNoiseSchedule,
    GammaNetwork,
    assert_mean_zero,
    remove_mean,
    write_to_csv,
)
import evaluation.diffusion_visualiser as vis
from evaluation.diffusion_analyze import (
    check_stability,
    sanitize_molecules_openbabel,
    sanitize_molecules_openbabel_batch,
)
from config_file import get_dataset_info
from evaluation.diffusion_distribution import (
    prepare_context,
    get_distributions,
)
from aqm.analyze_sampling import SamplingMetrics
from aqm.info_data import AQMInfos

from e3moldiffusion.molfeat import get_bond_feature_dims
from e3moldiffusion.coordsatoms import ScoreAtomsModel


def zero_mean(x: Tensor, batch: Tensor, dim_size: int, dim=0):
    out = x - scatter_mean(x, index=batch, dim=dim, dim_size=dim_size)[batch]
    return out


def assert_zero_mean(x: Tensor, batch: Tensor, dim_size: int, dim=0, eps: float = 1e-6):
    out = scatter_mean(x, index=batch, dim=dim, dim_size=dim_size).mean()
    return abs(out) < eps


class Trainer(pl.LightningModule):
    def __init__(
        self,
        hparams,
        dataset_info=None,
        dataset_statistics=None,
        train_smiles=None,
        properties_norm=None,
        nodes_dist=None,
        prop_dist=None,
        data_std: float = 0.5,
        epsilon_time: float = 0.002,
        T: float = 80.0,
        bins_min: int = 2,
        bins_max: int = 150,
        rho: float = 7,
        initial_ema_decay: float = 0.9,
    ):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.i = 0

        self.properties_norm = properties_norm

        self.dataset_info = dataset_info
        self.train_smiles = train_smiles
        self.nodes_dist = nodes_dist
        self.prop_dist = prop_dist

        self.val_sampling_metrics = SamplingMetrics(
            self.train_smiles, dataset_statistics, test=False
        )

        assert self.hparams.loss_type in {"vlb", "l2"}
        self.loss_type = self.hparams.loss_type

        self.include_charges = self.hparams.include_charges

        if self.hparams.noise_schedule == "learned":
            assert self.loss_type == "vlb", (
                "A noise schedule can only be learned" " with a vlb objective."
            )

        # Only supported parametrization.
        assert self.hparams.parametrization == "eps"

        if self.hparams.noise_schedule == "learned":
            self.gamma = GammaNetwork()
        else:
            self.gamma = PredefinedNoiseSchedule(
                self.hparams.noise_schedule,
                timesteps=self.hparams.num_diffusion_timesteps,
                precision=self.hparams.noise_precision,
            )

        self.num_atom_features = self.hparams.num_atom_types + int(self.include_charges)
        self.num_bond_classes = 5
        self.n_dims = 3  # positions are always 3D
        self.num_context_features = self.hparams.num_context_features
        self.num_classes = self.num_atom_features - int(self.include_charges)

        self.T = self.hparams.num_diffusion_timesteps

        self.parametrization = self.hparams.parametrization

        self.norm_values = self.hparams.norm_values
        self.norm_biases = self.hparams.norm_biases
        self.register_buffer("buffer", torch.zeros(1))

        if self.hparams.noise_schedule != "learned":
            self.check_issues_norm_values()

        if self.hparams.model == "eqgat":
            self.model = ScoreAtomsModel(
                hn_dim=(hparams["sdim"], hparams["vdim"]),
                num_layers=hparams["num_layers"],
                use_norm=hparams["use_norm"],
                use_cross_product=hparams["use_cross_product"],
                num_atom_features=self.num_atom_features,
                rbf_dim=hparams["num_rbf"],
                cutoff_local=hparams["cutoff_upper"],
                cutoff_global=hparams["cutoff_upper_global"],
                num_context_features=hparams["num_context_features"],
                vector_aggr="mean",
                local_global_model=hparams["fully_connected_layer"],
                fully_connected=hparams["fully_connected"],
            )
        else:
            raise ValueError(f"Unknown architecture: {self.hparams.model}")

        self.model_ema = copy.deepcopy(self.model)
        self.model_ema.requires_grad_(False)

        self.data_std = data_std
        self.epsilon_time = epsilon_time
        self.T = T
        self.rho = rho
        self.bins_min = bins_min
        self.bins_max = bins_max

        self.loss_fn = mse_loss

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

    def forward(
        self,
        z,
        pos,
        batch,
        times,
        context,
        edge_index_local,
    ):
        return self._forward(
            z=z,
            pos=pos,
            batch=batch,
            t=times,
            context=context,
            edge_index_local=edge_index_local,
            model=self.model,
        )

    def _forward(
        self,
        z,
        pos,
        batch,
        times,
        context,
        edge_index_local,
        model: nn.Module,
    ):
        skip_coef = self.data_std**2 / (
            (times - self.time_min).pow(2) + self.data_std**2
        )
        out_coef = self.data_std * times / (times.pow(2) + self.data_std**2).pow(0.5)

        out = model(
            z=z,
            t=times,
            pos=pos,
            batch=batch,
            context=context,
            edge_index_local=edge_index_local,
        )

        # TO-DO: remove mean, add skip_coef and out_coef, separate score_coords and score_atoms
        return out

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")

    def validation_step(self, batch, batch_idx, *args):
        if len(args) == 0 or (len(args) > 0 and args[0] == 0):
            # validation step
            return self.step(batch, "val")
        # test step
        return self.step(batch, "test")

    def test_step(self, batch, batch_idx):
        return self.step(batch, "test")

    def step(self, batch: torch.Tensor, stage: str):
        node_feat: Tensor = batch.xgeom
        pos: Tensor = batch.pos
        data_batch: Tensor = batch.batch
        bs = int(data_batch.max()) + 1

        _bins = self.bins
        timesteps = torch.randint(
            0,
            _bins - 1,
            (bs,),
            device=batch.batch.device,
        ).long()

        current_times = self.timesteps_embedding(timesteps, _bins)
        next_times = self.timesteps_embedding(timesteps + 1, _bins)

        # POSITION NOISING
        pos_centered = zero_mean(pos, data_batch, dim=0, dim_size=bs)

        noise_coords_t = torch.randn_like(pos) * current_times
        noise_coords_t = zero_mean(noise_coords_t, batch=data_batch, dim_size=bs, dim=0)
        noise_coords_t_next = torch.randn_like(pos) * next_times
        noise_coords_t_next = zero_mean(
            noise_coords_t_next, batch=data_batch, dim_size=bs, dim=0
        )

        pos_perturbed_t = pos_centered + current_times * noise_coords_t
        pos_perturbed_t_next = pos_centered + next_times * noise_coords_t_next

        # ONE-HOT ATOM NOISING
        xohe = F.one_hot(node_feat, num_classes=self.hparams.num_atom_types).float()
        xohe = 0.25 * xohe

        noise_ohes_t = torch.randn_like(xohe)
        mean_ohes_t, std_ohes_t = self.sde.marginal_prob(x=xohe, t=current_times)
        ohes_perturbed_t = mean_ohes_t + std_ohes_t * noise_ohes_t

        noise_ohes_t_next = torch.randn_like(xohe)
        mean_ohes_t_next, std_ohes_t_next = self.sde.marginal_prob(x=xohe, t=next_times)
        ohes_perturbed_t_next = mean_ohes_t_next + std_ohes_t_next * noise_ohes_t_next

        # BUILD GRAPHS
        edge_index_local_t = radius_graph(
            x=pos_perturbed_t,
            r=self.hparams.cutoff_local,
            batch=data_batch,
            max_num_neighbors=self.hparams.max_num_neighbors,
        )
        edge_index_local_t_next = radius_graph(
            x=pos_perturbed_t_next,
            r=self.hparams.cutoff_local,
            batch=data_batch,
            max_num_neighbors=self.hparams.max_num_neighbors,
        )

        # FORWARD THROUGH EMA MODEL for t_n
        with torch.no_grad():
            target = self._forward(
                model=self.model_ema,
                x=ohes_perturbed_t,
                t=current_times,
                pos=pos_perturbed_t,
                edge_index_local=edge_index_local_t,
                batch=data_batch,
            )
        # FORWARD THROUGH MODEL for t_(n+1)
        net_out = self(
            x=ohes_perturbed_t_next,
            t=next_times,
            pos=pos_perturbed_t_next,
            edge_index_local=edge_index_local_t_next,
            batch=data_batch,
        )
        coords_loss, ohes_loss = self.loss_fn(net_out, target)
        loss = coords_loss + ohes_loss

        self.log(
            f"{stage}/loss",
            loss,
            on_step=True,
            batch_size=bs,
        )
        self.log(
            f"{stage}/coords_loss",
            coords_loss,
            on_step=True,
            batch_size=bs,
        )
        self.log(
            f"{stage}/ohes_loss",
            ohes_loss,
            on_step=True,
            batch_size=bs,
        )
        self._bins_tracker(_bins)
        self.log(
            "bins",
            self._bins_tracker,
            on_step=True,
            on_epoch=False,
            logger=True,
        )
        return loss

    def optimizer_step(self, *args, **kwargs) -> None:
        super().optimizer_step(*args, **kwargs)
        self.ema_update()

    @torch.no_grad()
    def ema_update(self):
        param = [p.data for p in self.model.parameters()]
        param_ema = [p.data for p in self.model_ema.parameters()]

        torch._foreach_mul_(param_ema, self.ema_decay)
        torch._foreach_add_(param_ema, param, alpha=1 - self.ema_decay)

        self._ema_decay_tracker(self.ema_decay)
        self.log(
            "ema_decay",
            self._ema_decay_tracker,
            on_step=True,
            on_epoch=False,
            logger=True,
        )

    @property
    def ema_decay(self):
        return math.exp(self.bins_min * math.log(self.initial_ema_decay) / self.bins)

    @property
    def bins(self) -> int:
        return math.ceil(
            math.sqrt(
                self.trainer.global_step
                / self.trainer.estimated_stepping_batches
                * (self.bins_max**2 - self.bins_min**2)
                + self.bins_min**2
            )
        )

    def timesteps_embedding(self, timesteps: torch.LongTensor, bins: int):
        return (
            (
                self.epsilon ** (1 / self.rho)
                + timesteps
                / (bins - 1)
                * (self.T ** (1 / self.rho) - self.epsilon ** (1 / self.rho))
            )
            .pow(self.rho)
            .clamp(0, self.T)
        )


if __name__ == "__main__":
    from aqm.data import AQMDataModule
    from aqm.hparams import add_arguments

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
        monitor="val/loss",
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

    dataset_statistics = AQMInfos(datamodule.dataset)

    smiles = datamodule.dataset.data.smiles
    train_idx = [int(i) for i in datamodule.idx_train]
    train_smiles = [smi for i, smi in enumerate(smiles) if i in train_idx]
    dataset_info = get_dataset_info(hparams.dataset, hparams.remove_hs)

    properties_norm = None
    if len(hparams.properties_list) > 0:
        properties_norm = datamodule.compute_mean_mad(hparams.properties_list)

    dataloader = datamodule.get_dataloader(datamodule.train_dataset, "val")
    nodes_dist, prop_dist = get_distributions(hparams, dataset_info, dataloader)
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
