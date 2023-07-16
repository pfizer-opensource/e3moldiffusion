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
        save_top_k=1,
        monitor="val/coords_loss",
        save_last=True,
    )
    lr_logger = LearningRateMonitor()
    tb_logger = TensorBoardLogger(
        hparams.save_dir + f"/run{hparams.id}/", default_hp_metric=False
    )

    print(f"Loading {hparams.dataset} Datamodule.")
    if hparams.use_adaptive_loader:
        print("Using adaptive dataloader")
        from experiments.data.geom_dataset_adaptive import GeomDataModule

        datamodule = GeomDataModule(hparams)
    else:
        print("Using non-adaptive dataloader")
        from experiments.data.geom_dataset_nonadaptive import GeomDataModule

        datamodule = GeomDataModule(
            root=hparams.dataset_root,
            batch_size=hparams.batch_size,
            num_workers=hparams.num_workers,
            pin_memory=True,
            persistent_workers=True,
            with_hydrogen=not hparams.no_h,
        )
        datamodule.prepare_data()
        datamodule.setup("fit")

    if hparams.continuous:
        from experiments.diffusion_pretrain_continuous import Trainer

        model = Trainer(
            hparams=hparams.__dict__,
        )
    else:
        from experiments.diffusion_pretrain_discrete import Trainer

        # TO-DO!
        # model = Trainer(
        #     hparams=hparams.__dict__,
        #     dataset_info=dataset_info,
        #     dataset_statistics=dataset_statistics,
        #     smiles_list=list(train_smiles),
        # )

    strategy = "ddp" if hparams.gpus > 1 else "auto"

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
    )

    pl.seed_everything(seed=0, workers=hparams.gpus > 1)

    trainer.fit(
        model=model,
        datamodule=datamodule,
        ckpt_path=hparams.load_ckpt if hparams.load_ckpt != "" else None,
    )
