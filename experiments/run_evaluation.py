import warnings
import argparse
import torch
from experiments.data.distributions import DistributionProperty
from experiments.data.utils import write_xyz_file

import pickle
import os

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
    save_dir,
    save_xyz=False,
    ngraphs=5000,
    batch_size=80,
    step=0,
    ddpm=True,
    eta_ddim=1.0,
):
    # load hyperparameter
    hparams = torch.load(model_path)["hyper_parameters"]
    hparams = dotdict(hparams)

    hparams.load_ckpt_from_pretrained = False
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
        dataset = "drugs"  # take dataset infos from GEOM for simplicity
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

    from experiments.data.data_info import GeneralInfos as DataInfos

    dataset_info = DataInfos(datamodule, hparams)

    train_smiles = (
        list(datamodule.train_dataset.smiles) if hparams.dataset != "pubchem" else None
    )
    prop_norm, prop_dist = None, None
    if len(hparams.properties_list) > 0 and hparams.context_mapping:
        prop_norm = datamodule.compute_mean_mad(hparams.properties_list)
        prop_dist = DistributionProperty(datamodule, hparams.properties_list)
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
        run_evaluation=True,
        strict=False,
    ).to(device)
    model = model.eval()

    results_dict, generated_smiles, stable_molecules = model.run_evaluation(
        step=step,
        device="cpu", # for storing metric tensors
        dataset_info=model.dataset_info,
        ngraphs=ngraphs,
        bs=batch_size,
        return_molecules=True,
        verbose=True,
        inner_verbose=True,
        save_dir=save_dir,
        ddpm=ddpm,
        eta_ddim=eta_ddim,
        run_test_eval=True,
    )

    if save_xyz:
        context = []
        atom_decoder = stable_molecules[0].dataset_info.atom_decoder
        for i in range(len(stable_molecules)):
            types = [atom_decoder[int(a)] for a in stable_molecules[i].atom_types]
            write_xyz_file(
                stable_molecules[i].positions,
                types,
                os.path.join(save_dir, f"mol_{i}.xyz"),
            )
            if prop_dist is not None:
                tmp = []
                for j, key in enumerate(hparams.properties_list):
                    mean, mad = (
                        prop_dist.normalizer[key]["mean"],
                        prop_dist.normalizer[key]["mad"],
                    )
                    prop = stable_molecules[i].context[j] * mad + mean
                    tmp.append(float(prop))
                context.append(tmp)

    if prop_dist is not None and save_xyz:
        with open(os.path.join(save_dir, "context.pickle"), "wb") as f:
            pickle.dump(context, f)
    with open(os.path.join(save_dir, "generated_smiles.pickle"), "wb") as f:
        pickle.dump(generated_smiles, f)
    with open(os.path.join(save_dir, "stable_molecules.pickle"), "wb") as f:
        pickle.dump(stable_molecules, f)


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description='Data generation')
    parser.add_argument('--model-path', default="/hpfs/userws/cremej01/workspace/logs/aqm_qm7x/x0_t_weighting_dip_mpol/best_mol_stab.ckpt", type=str,
                        help='Path to trained model')
    parser.add_argument('--save-dir', default="/hpfs/userws/cremej01/workspace/logs/aqm_qm7x/x0_t_weighting_dip_mpol", type=str,
                        help='Path to test output')
    parser.add_argument('--save-xyz', default=False, action="store_true",
                        help='Whether or not to store generated molecules in xyz files')
    parser.add_argument('--ngraphs', default=5000, type=int,
                            help='How many graphs to sample. Defaults to 5000')
    parser.add_argument('--batch-size', default=80, type=int,
                            help='Batch-size to generate the selected ngraphs. Defaults to 80.')
    parser.add_argument('--ddim', default=False, action="store_true",
                        help='If DDIM sampling should be used. Defaults to False')
    parser.add_argument('--eta-ddim', default=1.0, type=float,
                        help='How to scale the std of noise in the reverse posterior. \
                            Can also be used for DDPM to track a deterministic trajectory. \
                            Defaults to 1.0')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    # Evaluate negative log-likelihood for the test partitions
    evaluate(
        model_path=args.model_path,
        save_dir=args.save_dir,
        ngraphs=args.ngraphs,
        batch_size=args.batch_size,
        ddpm=not args.ddim,
        eta_ddim=args.eta_ddim,
    )
