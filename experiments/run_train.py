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
from experiments.data.distributions import DistributionProperty

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
    non_adaptive = True
    if hparams.dataset == "drugs":
        dataset = "drugs"
        if hparams.use_adaptive_loader:
            print("Using adaptive dataloader")
            non_adaptive = False
            from experiments.data.geom.geom_dataset_adaptive import (
                GeomDataModule as DataModule,
            )
        else:
            print("Using non-adaptive dataloader")
            from experiments.data.geom.geom_dataset_nonadaptive import (
                GeomDataModule as DataModule,
            )
    elif hparams.dataset == "qm9":
        dataset = "qm9"
        from experiments.data.qm9.qm9_dataset import QM9DataModule as DataModule

    elif hparams.dataset == "aqm":
        dataset = "aqm"
        from experiments.data.aqm.aqm_dataset_nonadaptive import (
            AQMDataModule as DataModule,
        )

    elif hparams.dataset == "aqm_qm7x":
        dataset = "aqm_qm7x"
        from experiments.data.aqm_qm7x.aqm_qm7x_dataset_nonadaptive import (
            AQMQM7XDataModule as DataModule,
        )
    elif hparams.dataset == "pcqm4mv2":
        dataset = "pcqm4mv2"
        if hparams.use_adaptive_loader:
            print("Using adaptive dataloader")
            from experiments.data.pcqm4mv2.pcqm4mv2_dataset_adaptive import (
                PCQM4Mv2DataModule as DataModule,
            )
        else:
            print("Using non-adaptive dataloader")
            from experiments.data.pcqm4mv2.pcqm4mv2_dataset_nonadaptive import (
                PCQM4Mv2DataModule as DataModule,
            )
    elif hparams.dataset == "pubchem":
        dataset = "pubchem"  # take dataset infos from GEOM for simplicity
        if hparams.use_adaptive_loader:
            print("Using adaptive dataloader")
            non_adaptive = False
            from experiments.data.pubchem.pubchem_dataset_adaptive import (
                PubChemDataModule as DataModule,
            )
        else:
            print("Using non-adaptive dataloader")
            from experiments.data.pubchem.pubchem_dataset_nonadaptive import (
                PubChemDataModule as DataModule,
            )
    elif hparams.dataset == "cross_docked":
        from experiments.data.ligand.ligand_dataset_nonadaptive import (
            LigandPocketDataModule as DataModule,
        )

    datamodule = DataModule(hparams)
    if non_adaptive:
        datamodule.prepare_data()
        datamodule.setup("fit")

    from experiments.data.data_info import GeneralInfos as DataInfos

    if dataset == "aqm_qm7x":
        from experiments.data.aqm.aqm_dataset_nonadaptive import (
            AQMDataModule as DataModule,
        )

        datamodule_aqm = DataModule(hparams)
        if non_adaptive:
            datamodule_aqm.prepare_data()
        dataset_info = DataInfos(datamodule_aqm, hparams)
        del datamodule_aqm
    else:
        dataset_info = DataInfos(datamodule, hparams)

    train_smiles = (
        (
            list(datamodule.train_dataset.smiles)
            if hparams.dataset != "pubchem"
            else None
        )
        if not hparams.select_train_subset
        else datamodule.train_smiles
    )
    prop_norm, prop_dist = None, None
    if len(hparams.properties_list) > 0 and hparams.context_mapping:
        prop_norm = datamodule.compute_mean_mad(hparams.properties_list)
        prop_dist = DistributionProperty(datamodule, hparams.properties_list)
        prop_dist.set_normalizer(prop_norm)

    if hparams.continuous:
        print("Using continuous diffusion")
        if hparams.diffusion_pretraining:
            print("Starting pre-training")
            from experiments.diffusion_pretrain_continuous import Trainer
        else:
            from experiments.diffusion_continuous import Trainer
    elif hparams.bond_prediction:
        print("Starting bond prediction model via discrete diffusion")
        from experiments.diffusion_discrete import Trainer
    elif hparams.property_prediction:
        print("Starting property prediction model via discrete diffusion")
        from experiments.diffusion_discrete import Trainer
    elif hparams.latent_dim:
        print("Using latent diffusion")
        from experiments.diffusion_latent_discrete import Trainer
    else:
        print("Using discrete diffusion")
        if hparams.diffusion_pretraining:
            print("Starting pre-training")
            if hparams.additional_feats:
                from experiments.diffusion_pretrain_discrete_moreFeats import Trainer
            else:
                from experiments.diffusion_pretrain_discrete import Trainer
        elif hparams.additional_feats:
            print("Using additional features")
            from experiments.diffusion_discrete_moreFeats import Trainer
        else:
            from experiments.diffusion_discrete import Trainer

    model = Trainer(
        hparams=hparams.__dict__,
        dataset_info=dataset_info,
        smiles_list=train_smiles,
        prop_dist=prop_dist,
        prop_norm=prop_norm,
    )

    from pytorch_lightning.plugins.environments import LightningEnvironment

    strategy = "ddp" if hparams.gpus > 1 else "auto"
    callbacks = [
            ema_callback,
            lr_logger,
            checkpoint_callback,
            TQDMProgressBar(refresh_rate=5),
            ModelSummary(max_depth=2),
        ]
    
    if hparams.ema_decay == 1.0:
        callbacks = callbacks[1:]
    
    trainer = pl.Trainer(
        accelerator="gpu" if hparams.gpus else "cpu",
        devices=hparams.gpus if hparams.gpus else 1,
        strategy=strategy,
        plugins=LightningEnvironment(),
        num_nodes=1,
        logger=tb_logger,
        enable_checkpointing=True,
        accumulate_grad_batches=hparams.accum_batch,
        val_check_interval=hparams.eval_freq,
        gradient_clip_val=hparams.grad_clip_val,
        callbacks=callbacks,
        precision=hparams.precision,
        num_sanity_val_steps=2,
        max_epochs=hparams.num_epochs,
        detect_anomaly=hparams.detect_anomaly,
    )

    pl.seed_everything(seed=hparams.seed, workers=hparams.gpus > 1)

    trainer.fit(
        model=model,
        datamodule=datamodule,
        ckpt_path=hparams.load_ckpt if hparams.load_ckpt != "" else None,
    )
