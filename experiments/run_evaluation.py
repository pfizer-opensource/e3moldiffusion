import warnings
import argparse
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
    model_path, save_dir, ngraphs=5000, batch_size=80, step=0, ddpm=True, eta_ddim=1.0
):
    # load hyperparameter
    hparams = torch.load(model_path)["hyper_parameters"]
    hparams = dotdict(hparams)

    print(f"Loading {hparams.dataset} Datamodule.")
    non_adaptive = True
    if hparams.dataset == "drugs":
        dataset = "drugs"
        from experiments.data.data_info import GEOMInfos as DataInfos

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
        from experiments.data.data_info import QM9Infos as DataInfos
        from experiments.data.qm9.qm9_dataset import QM9DataModule as DataModule

    elif hparams.dataset == "pubchem":
        dataset = "drugs"  # take dataset infos from GEOM for simplicity
        from experiments.data.data_info import PubChemInfos as DataInfos

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

    datamodule = DataModule(hparams)
    if non_adaptive:
        datamodule.prepare_data()
        datamodule.setup("fit")

    dataset_info = DataInfos(datamodule, hparams)

    # temporary
    from experiments.data.config_file import get_dataset_info
    from experiments.utils import get_empirical_num_nodes

    dataset_i = get_dataset_info(hparams.dataset, hparams.remove_hs)
    empirical_num_nodes = get_empirical_num_nodes(dataset_i)

    train_smiles = datamodule.train_dataset.smiles

    prop_norm, prop_dist = None, None
    properties_list = []
    context_mapping = False
    if len(properties_list) > 0 and context_mapping:
        dataloader = datamodule.get_dataloader(datamodule.train_dataset, "val")
        prop_norm = datamodule.compute_mean_mad(hparams.properties_list)
        prop_dist = DistributionProperty(dataloader, hparams.properties_list)
        prop_dist.set_normalizer(prop_norm)

    if hparams.continuous:
        print("Using continuous diffusion")
        from experiments.diffusion_continuous import Trainer
    elif hparams.bond_prediction:
        print("Starting bond prediction model via discrete diffusion")
        from experiments.bond_prediction_discrete import Trainer
    elif hparams.latent_dim:
        print("Using latent diffusion")
        # from experiments.diffusion_latent_discrete import Trainer #need refactor
        raise NotImplementedError
    else:
        print("Using discrete diffusion")
        if hparams.additional_feats:
            print("Using additional features")
            from experiments.diffusion_discrete_moreFeats import Trainer
        else:
            from experiments.diffusion_discrete import Trainer

    # if you want bond_model_guidance, flag this here in the Trainer
    device = "cuda"
    model = Trainer.load_from_checkpoint(
        model_path,
        dataset_info=dataset_info,
        smiles_list=list(train_smiles),
        prop_norm=prop_norm,
        prop_dist=prop_dist,
        # context_mapping=False,
        # num_context_features=0,
        # empirical_num_nodes=empirical_num_nodes,
        # bond_model_guidance=False,
        strict=False,
    ).to(device)
    model = model.eval()

    results_dict, generated_smiles = model.run_evaluation(
        step=step,
        dataset_info=model.dataset_info,
        ngraphs=ngraphs,
        bs=batch_size,
        return_smiles=True,
        verbose=True,
        inner_verbose=True,
        save_dir=save_dir,
        ddpm=ddpm,
        eta_ddim=eta_ddim,
    )
    return results_dict, generated_smiles


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description='Data generation')
    parser.add_argument('--model-path', default="/hpfs/userws/cremej01/projects/logs/geom/adaptive/run0/last.ckpt", type=str,
                        help='Path to trained model')
    parser.add_argument('--save-dir', default="/hpfs/userws/cremej01/projects/logs/geom/evaluation", type=str,
                        help='Path to test output')
    parser.add_argument('--ngraphs', default=5000, type=int,
                            help='How many graphs to sample. Defaults to 5000')
    parser.add_argument('--batch_size', default=80, type=int,
                            help='Batch-size to generate the selected ngraphs. Defaults to 80.')
    parser.add_argument('--ddim', default=False, action="store_true",
                        help='If DDIM sampling should be used. Defaults to False')
    parser.add_argument('--eta_ddim', default=1.0, type=float,
                        help='How to scale the std of noise in the reverse posterior. \
                            Can also be used for DDPM to track a deterministic trajectory. \
                            Defaults to 1.0')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    # Evaluate negative log-likelihood for the test partitions
    results_dict, generated_smiles = evaluate(
        model_path=args.model_path,
        save_dir=args.save_dir,
        ngraphs=args.ngraphs,
        batch_size=args.batch_size,
        ddpm=not args.ddim,
        eta_ddim=args.eta_ddim,
    )
