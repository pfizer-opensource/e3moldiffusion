import os
from callbacks.ema import ExponentialMovingAverage
from argparse import ArgumentParser
import pytorch_lightning as pl
import torch.nn.functional as F
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger
import warnings
from experiments.data.data_info import PubChemInfos

warnings.filterwarnings(
    "ignore", category=UserWarning, message="TypedStorage is deprecated"
)

from experiments.hparams import add_arguments

if __name__ == "__main__":
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
        save_top_k=3,
        monitor="val/loss",
        save_last=True,
    )
    lr_logger = LearningRateMonitor()
    tb_logger = TensorBoardLogger(
        hparams.save_dir + f"/run{hparams.id}/", default_hp_metric=False
    )

    print(f"Loading {hparams.dataset} Datamodule.")
    if hparams.use_adaptive_loader:
        print("Using adaptive dataloader")
        from experiments.data.pubchem.pubchem_dataset_adaptive import PubChemDataModule

        datamodule = PubChemDataModule(hparams)
    else:
        print("Using non-adaptive dataloader")
        from experiments.data.pubchem.pubchem_dataset_nonadaptive import (
            PubChemDataModule,
        )

        datamodule = PubChemDataModule(
            hparams=hparams,
        )
        datamodule.prepare_data()
        datamodule.setup("fit")

    dataset_info = PubChemInfos(datamodule, hparams)

    if hparams.continuous:
        from experiments.deprecated.diffusion_pretrain_continuous import Trainer

        model = Trainer(
            hparams=hparams.__dict__,
        )
    else:
        from experiments.deprecated.diffusion_pretrain_discrete import Trainer

        model = Trainer(
            hparams=hparams.__dict__,
            dataset_info=dataset_info,
            smiles_list=None,
        )

    from pytorch_lightning.plugins.environments import LightningEnvironment
    strategy = "ddp" if hparams.gpus > 1 else "auto"    
    trainer = pl.Trainer(
        accelerator="gpu" if hparams.gpus else "cpu",
        devices=hparams.gpus if hparams.gpus else None,
        strategy=strategy,
        plugins=LightningEnvironment(),
        num_nodes=1,
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
    )

    pl.seed_everything(seed=0, workers=hparams.gpus > 1)

    trainer.fit(
        model=model,
        datamodule=datamodule,
        ckpt_path=hparams.load_ckpt if hparams.load_ckpt != "" else None,
    )
