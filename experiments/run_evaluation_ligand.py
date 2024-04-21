import argparse
import os
import warnings

import numpy as np
import pytorch_lightning as pl
import torch

from experiments.data.distributions import DistributionProperty

warnings.filterwarnings(
    "ignore", category=UserWarning, message="TypedStorage is deprecated"
)
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
    save_xyz=True,
    calculate_energy=False,
    batch_size=2,
    use_ligand_dataset_sizes=False,
    prior_n_atoms="targetdiff",
    n_nodes_bias=2,
    dataset_root=None,
    build_obabel_mol=False,
    save_traj=False,
    property_classifier_guidance=False,
    property_classifier_self_guidance=False,
    property_classifier_guidance_complex=False,
    classifier_guidance_scale=1.0e-4,
    ckpt_property_model=None,
    ddpm=True,
    eta_ddim=1.0,
):
    print("Loading from checkpoint; adapting hyperparameters to specified args")

    # load model
    ckpt = torch.load(model_path)
    ckpt["hyper_parameters"]["load_ckpt"] = None
    ckpt["hyper_parameters"]["load_ckpt_from_pretrained"] = None
    ckpt["hyper_parameters"]["test_save_dir"] = save_dir
    ckpt["hyper_parameters"]["calculate_energy"] = calculate_energy
    ckpt["hyper_parameters"]["save_xyz"] = save_xyz
    ckpt["hyper_parameters"]["batch_size"] = batch_size
    ckpt["hyper_parameters"]["select_train_subset"] = False
    ckpt["hyper_parameters"]["diffusion_pretraining"] = False
    ckpt["hyper_parameters"]["gpus"] = 1
    ckpt["hyper_parameters"]["use_ligand_dataset_sizes"] = use_ligand_dataset_sizes
    ckpt["hyper_parameters"]["build_obabel_mol"] = build_obabel_mol
    ckpt["hyper_parameters"]["save_traj"] = save_traj
    ckpt["hyper_parameters"]["num_charge_classes"] = 6
    ckpt["hyper_parameters"][
        "property_classifier_guidance"
    ] = property_classifier_guidance
    ckpt["hyper_parameters"][
        "property_classifier_self_guidance"
    ] = property_classifier_self_guidance
    ckpt["hyper_parameters"][
        "property_classifier_guidance_complex"
    ] = property_classifier_guidance_complex
    ckpt["hyper_parameters"]["classifier_guidance_scale"] = classifier_guidance_scale
    ckpt["hyper_parameters"]["ckpt_property_model"] = ckpt_property_model
    ckpt["hyper_parameters"]["n_nodes_bias"] = n_nodes_bias
    ckpt["hyper_parameters"]["prior_n_atoms"] = prior_n_atoms
    ckpt["hyper_parameters"]["dataset_root"] = dataset_root
    ckpt["hyper_parameters"]["ckpt_property_model"] = ckpt_property_model
    ckpt["hyper_parameters"]["ddpm"] = ddpm
    ckpt["hyper_parameters"]["eta_ddim"] = eta_ddim

    ckpt_path = os.path.join(save_dir, "test_model.ckpt")
    torch.save(ckpt, ckpt_path)

    hparams = ckpt["hyper_parameters"]
    hparams = dotdict(hparams)

    print(f"Loading {hparams.dataset} Datamodule.")
    dataset = "crossdocked"
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
    print(f"Loading {hparams.dataset} Datamodule.")
    if hparams.dataset == "crossdocked":
        dataset = "crossdocked"
        if hparams.use_adaptive_loader:
            print(f"Using non-adaptive dataloader for {dataset}")
            from experiments.data.ligand.ligand_dataset_adaptive import (
                LigandPocketDataModule as DataModule,
            )
        else:
            print("Using non-adaptive dataloader")
            from experiments.data.ligand.ligand_dataset_nonadaptive import (
                LigandPocketDataModule as DataModule,
            )
    elif hparams.dataset == "kinodata":
        dataset = hparams.dataset
        print(f"Using non-adaptive dataloader for {dataset}")
        from experiments.data.ligand.kino_dataset_nonadaptive import (
            LigandPocketDataModule as DataModule,
        )
    else:
        raise Exception("Dataset not available!")

    datamodule = DataModule(hparams)

    from experiments.data.data_info import GeneralInfos as DataInfos

    dataset_info = DataInfos(datamodule, hparams)
    histogram = os.path.join(hparams.dataset_root, "size_distribution.npy")
    histogram = np.load(histogram).tolist()
    test_smiles = list(datamodule.test_dataset.smiles)

    prop_norm, prop_dist = None, None
    if (
        len(hparams.properties_list) > 0
        and hparams.context_mapping
        and not hparams.use_centroid_context_embed
    ) or (
        hparams.property_training
        and not (
            hparams.regression_property == "sa_score"
            or hparams.regression_property == "docking_score"
            or hparams.regression_property == "ic50"
        )
    ):
        prop_norm = datamodule.compute_mean_mad(hparams.properties_list)
        prop_dist = DistributionProperty(datamodule, hparams.properties_list)
        prop_dist.set_normalizer(prop_norm)

    if hparams.additional_feats:
        from experiments.diffusion_discrete_pocket_addfeats import Trainer

        # from experiments.diffusion_discrete_pocket_addfeats_reduced import Trainer
    else:
        from experiments.diffusion_discrete_pocket import Trainer

    torch.cuda.empty_cache()

    # backward compatibility
    if "use_centroid_context_embed" not in hparams:
        hparams.use_centroid_context_embed = False
    if "use_latent_encoder" not in hparams:
        hparams.use_latent_encoder = False
    if "use_scaffold_latent_embed" not in hparams:
        hparams.use_scaffold_latent_embed = False
    if "flow_matching" not in hparams:
        hparams.flow_matching = False

    if build_obabel_mol:
        print(
            "Sampled molecules will be built with OpenBabel (without bond information)!"
        )

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
        smiles_list=test_smiles,
        prop_dist=prop_dist,
        prop_norm=prop_norm,
        histogram=histogram,
    )

    trainer.test(model, datamodule=datamodule, ckpt_path=ckpt_path)


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description='Data generation')
    parser.add_argument('--model-path', default="/hpfs/userws/cremej01/workspace/logs/aqm_qm7x/x0_t_weighting_dip_mpol/best_mol_stab.ckpt", type=str,
                        help='Path to trained model')
    parser.add_argument('--dataset-root', default=None, type=str, help='Path to data')
    parser.add_argument("--property-classifier-guidance", default=False, action="store_true")
    parser.add_argument("--property-classifier-guidance-complex", default=False, action="store_true")
    parser.add_argument("--property-classifier-self-guidance", default=False, action="store_true")
    parser.add_argument("--ckpt-property-model", default=None, type=str)
    parser.add_argument("--prior-n-atoms", default="targetdiff", type=str)
    parser.add_argument("--n-nodes-bias", default=2, type=int)
    parser.add_argument('--classifier-guidance-scale', default=1.0e-4, type=float,
                        help='How to scale the guidance shift')
    parser.add_argument("--use-ligand-dataset-sizes", default=False, action="store_true")
    parser.add_argument("--build-obabel-mol", default=False, action="store_true")
    parser.add_argument("--save-traj", default=False, action="store_true")
    parser.add_argument('--save-dir', default="/hpfs/userws/cremej01/workspace/logs/aqm_qm7x/x0_t_weighting_dip_mpol", type=str,
                        help='Path to test output')
    parser.add_argument('--save-xyz', default=False, action="store_true",
                        help='Whether or not to store generated molecules in xyz files')
    parser.add_argument('--calculate-energy', default=False, action="store_true",
                        help='Whether or not to calculate xTB energies and forces')
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
        batch_size=args.batch_size,
        ddpm=not args.ddim,
        eta_ddim=args.eta_ddim,
        save_xyz=args.save_xyz,
        save_traj=args.save_traj,
        calculate_energy=args.calculate_energy,
        use_ligand_dataset_sizes=args.use_ligand_dataset_sizes,
        prior_n_atoms=args.prior_n_atoms,
        n_nodes_bias=args.n_nodes_bias,
        dataset_root=args.dataset_root,
        build_obabel_mol=args.build_obabel_mol,
        property_classifier_guidance=args.property_classifier_guidance,
        property_classifier_self_guidance=args.property_classifier_self_guidance,
        property_classifier_guidance_complex=args.property_classifier_guidance_complex,
        classifier_guidance_scale=args.classifier_guidance_scale,
        ckpt_property_model=args.ckpt_property_model,
    )
