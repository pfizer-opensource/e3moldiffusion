import argparse
import warnings

import pytorch_lightning as pl
import torch

from experiments.data.distributions import DistributionProperty

warnings.filterwarnings(
    "ignore", category=UserWarning, message="TypedStorage is deprecated"
)


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def evaluate(
    model_path,
):
    # load hyperparameter
    hparams = torch.load(model_path)["hyper_parameters"]
    hparams = dotdict(hparams)

    hparams.load_ckpt_from_pretrained = None
    hparams.load_ckpt = None
    hparams.gpus = 1

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
    elif hparams.dataset == "geomqm":
        dataset = "geomqm"
        from experiments.data.geom.geom_dataset_adaptive_qm import (
            GeomQMDataModule as DataModule,
        )
    else:
        raise Exception("Dataset not specified!")

    if dataset == "pubchem":
        datamodule = DataModule(hparams, evaluation=True)
    else:
        datamodule = DataModule(hparams)

    from experiments.data.data_info import GeneralInfos as DataInfos

    dataset_info = DataInfos(datamodule, hparams)

    train_smiles = (
        list(datamodule.train_dataset.smiles)
        if hparams.dataset != "pubchem"
        else datamodule.train_smiles
    )
    prop_norm, prop_dist = None, None
    prop_norm = datamodule.compute_mean_mad(hparams.properties_list)
    prop_dist = DistributionProperty(datamodule, hparams.properties_list)
    prop_dist.set_normalizer(prop_norm)

    if hparams.energy_training:
        print("Running energy model testing")
        assert hparams.dataset == "drugs"
        from experiments.energy_training import Trainer
    else:
        print(f"Running {hparams.regression_property} model testing")
        assert hparams.dataset == "geomqm"
        from experiments.property_training import Trainer

    trainer = pl.Trainer(
        accelerator="gpu" if hparams.gpus else "cpu",
        devices=1,
        strategy="auto",
        num_nodes=1,
        precision=hparams.precision,
    )
    pl.seed_everything(seed=hparams.seed, workers=hparams.gpus > 1)

    model = Trainer(
        hparams=hparams,
        dataset_info=dataset_info,
        smiles_list=train_smiles,
        prop_dist=prop_dist,
        prop_norm=prop_norm,
    )

    trainer.test(model, datamodule=datamodule, ckpt_path=model_path)


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description='Data generation')
    parser.add_argument('--model-path', default="/hpfs/userws/cremej01/workspace/logs/aqm_qm7x/x0_t_weighting_dip_mpol/best_mol_stab.ckpt", type=str,
                        help='Path to trained model')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    # Evaluate negative log-likelihood for the test partitions
    evaluate(
        model_path=args.model_path,
    )
