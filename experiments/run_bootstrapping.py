import argparse
import os
import warnings

import numpy as np
import pytorch_lightning as pl
from callbacks.ema import ExponentialMovingAverage
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger

from experiments.data.distributions import DistributionProperty

warnings.filterwarnings(
    "ignore", category=UserWarning, message="TypedStorage is deprecated"
)
warnings.filterwarnings("ignore", category=DeprecationWarning)


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def bootstrap(args):
    # load hyperparameter
    import torch

    ckpt = torch.load(args.model_path)
    ckpt["hyper_parameters"]["docking_scores_threshold"] = args.docking_scores_threshold
    ckpt["hyper_parameters"]["sascores_threshold"] = args.sascores_threshold
    ckpt["hyper_parameters"]["lr"] = args.lr
    ckpt["hyper_parameters"]["bootstrap_save_dir"] = args.save_dir
    ckpt["hyper_parameters"]["batch_size"] = args.batch_size
    ckpt["hyper_parameters"]["inference_batch_size"] = args.inference_batch_size
    ckpt["hyper_parameters"]["gpus"] = args.gpus
    ckpt["hyper_parameters"]["property_guidance"] = args.property_guidance
    ckpt["hyper_parameters"]["ckpt_property_model"] = args.ckpt_property_model
    ckpt["hyper_parameters"]["guidance_scale"] = args.guidance_scale
    ckpt["hyper_parameters"]["fix_n_nodes"] = args.fix_n_nodes
    ckpt["hyper_parameters"]["vary_n_nodes"] = args.vary_n_nodes
    ckpt["hyper_parameters"]["n_nodes_bias"] = args.n_nodes_bias
    ckpt["hyper_parameters"]["num_ligands_per_pocket"] = args.num_ligands_per_pocket
    ckpt["hyper_parameters"]["select_train_subset"] = False
    ckpt["hyper_parameters"]["diffusion_pretraining"] = False
    ckpt["hyper_parameters"]["load_ckpt_from_pretrained"] = None
    ckpt["hyper_parameters"]["regression_property"] = ["docking_score", "sa_score"]
    ckpt["hyper_parameters"]["joint_property_prediction"] = True

    ckpt["epoch"] = 0
    ckpt["global_step"] = 0

    if ckpt["optimizer_states"][0]["param_groups"][0]["lr"] != args.lr:
        print("Changing learning rate ...")
        ckpt["optimizer_states"][0]["param_groups"][0]["lr"] = args.lr
        ckpt["optimizer_states"][0]["param_groups"][0]["initial_lr"] = args.lr

    ckpt_path = os.path.join(args.save_dir, f"bootstrap_model_lr{args.lr}.ckpt")
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    torch.save(ckpt, ckpt_path)

    hparams = ckpt["hyper_parameters"]
    hparams = dotdict(hparams)

    print(f"Loading {hparams.dataset} Datamodule.")
    if hparams.use_adaptive_loader:
        print("Using adaptive dataloader")
        from experiments.data.ligand.ligand_dataset_adaptive import (
            LigandPocketDataModule as DataModule,
        )
    else:
        print("Using non-adaptive dataloader")
        from experiments.data.ligand.ligand_dataset_nonadaptive import (
            LigandPocketDataModule as DataModule,
        )

    datamodule = DataModule(hparams, bootstrapping=True)

    from experiments.data.data_info import GeneralInfos as DataInfos

    dataset_info = DataInfos(datamodule, hparams)
    histogram = os.path.join(hparams.dataset_root, "size_distribution.npy")
    histogram = np.load(histogram).tolist()
    train_smiles = list(datamodule.train_dataset.smiles)

    prop_norm, prop_dist = None, None
    if (
        len(hparams.properties_list) > 0
        and hparams.context_mapping
        and not hparams.use_centroid_context_embed
    ) or (
        hparams.property_training
        and not (
            hparams.regression_property == "sascore"
            or hparams.regression_property == "docking_score"
        )
    ):
        prop_norm = datamodule.compute_mean_mad(hparams.properties_list)
        prop_dist = DistributionProperty(datamodule, hparams.properties_list)
        prop_dist.set_normalizer(prop_norm)

    from experiments.diffusion_bootstrapping import Trainer

    torch.cuda.empty_cache()

    model = Trainer(
        hparams=hparams,
        dataset_info=dataset_info,
        smiles_list=train_smiles,
        prop_dist=prop_dist,
        prop_norm=prop_norm,
        histogram=histogram,
    )

    from pytorch_lightning.plugins.environments import LightningEnvironment

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
    strategy = "ddp" if hparams.gpus > 1 else "auto"
    # strategy = 'ddp_find_unused_parameters_true'
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
        accelerator="gpu" if args.gpus else "cpu",
        devices=args.gpus if args.gpus else 1,
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
        reload_dataloaders_every_n_epochs=1,
        detect_anomaly=hparams.detect_anomaly,
    )

    pl.seed_everything(seed=hparams.seed, workers=args.gpus > 1)

    trainer.fit(
        model=model,
        # datamodule=datamodule,
        ckpt_path=ckpt_path,
    )


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description='Data generation')
    parser.add_argument('--model-path', default="/hpfs/userws/cremej01/workspace/logs/aqm_qm7x/x0_t_weighting_dip_mpol/best_mol_stab.ckpt", type=str,
                        help='Path to trained model')
    parser.add_argument('--dataset-root', default=None, type=str, help='Path to dataset')
    parser.add_argument("--gpus", default=1, type=int)
    parser.add_argument("--docking-scores-threshold", default=-7.0, type=float)
    parser.add_argument("--sascores-threshold", default=0.5, type=float)
    parser.add_argument("--lr", default=1.e-4, type=float)
    parser.add_argument("--fix-n-nodes", action="store_true")
    parser.add_argument("--vary-n-nodes", action="store_true")
    parser.add_argument("--n-nodes-bias", default=0, type=int)
    parser.add_argument("--property-guidance", default=False, action="store_true")
    parser.add_argument("--ckpt-property-model", default=None, type=str)
    parser.add_argument('--guidance-scale', default=1.0e-4, type=float,
                        help='How to scale the guidance shift')
    parser.add_argument('--save-dir', default="/scratch1/e3moldiffusion/logs/crossdocked/bootstrapping", type=str,
                        help='Path to test output')
    parser.add_argument('--batch-size', default=8, type=int,
                            help='Batch-size to generate the selected ngraphs. Defaults to 8.')
    parser.add_argument('--num-ligands-per-pocket', default=50, type=int,
                            help='Batch-size to generate the selected ngraphs. Defaults to 50.')
    parser.add_argument('--inference-batch-size', default=50, type=int,
                            help='Inference batch-size to generate the selected ngraphs. Defaults to 50.')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    # Evaluate negative log-likelihood for the test partitions
    bootstrap(args)
