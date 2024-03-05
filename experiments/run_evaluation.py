import argparse
import os
import pickle
import warnings
import json

import numpy as np
import torch
from tqdm import tqdm

from experiments.data.distributions import DistributionProperty
from experiments.data.utils import write_xyz_file
from experiments.xtb_energy import calculate_xtb_energy
from experiments.xtb_wrapper import xtb_calculate

warnings.filterwarnings(
    "ignore", category=UserWarning, message="TypedStorage is deprecated"
)


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def evaluate(args):
    # load hyperparameter
    hparams = torch.load(args.model_path)["hyper_parameters"]
    hparams["select_train_subset"] = False
    hparams["diffusion_pretraining"] = False
    hparams["num_charge_classes"] = 6
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
    elif hparams.dataset == "geomqm":
        dataset = "geomqm"
        from experiments.data.geom.geom_dataset_adaptive_qm import (
            GeomQMDataModule as DataModule,
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
    if (
        len(hparams.properties_list) > 0
        and hparams.context_mapping
        and not hparams.use_centroid_context_embed
    ) or (
        hparams.property_training
        and not (
            "sa_score" in hparams.regression_property
            or "docking_score" in hparams.regression_property
        )
        or hparams.joint_property_prediction
        and not (
            "sa_score" in hparams.regression_property
            or "docking_score" in hparams.regression_property
        )
    ):
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
        args.model_path,
        dataset_info=dataset_info,
        smiles_list=train_smiles,
        prop_norm=prop_norm,
        prop_dist=prop_dist,
        load_ckpt_from_pretrained=None,
        load_ckpt=None,
        run_evaluation=True,
        strict=False,
    ).to(device)
    model = model.eval()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if args.sample_only_valid:
        print("\nStarting sampling of only valid molecules...\n")
        results_dict, generated_smiles, stable_molecules = model.generate_valid_samples(
            dataset_info=model.dataset_info,
            ngraphs=args.ngraphs,
            bs=args.batch_size,
            return_molecules=True,
            verbose=True,
            inner_verbose=True,
            save_dir=args.save_dir,
            ddpm=not args.ddim,
            eta_ddim=args.eta_ddim,
            save_traj=args.save_traj,
            fix_noise_and_nodes=args.fix_noise_and_nodes,
            n_nodes=args.n_nodes,
            vary_n_nodes=args.vary_n_nodes,
            relax_sampling=args.relax_sampling,
            relax_steps=args.relax_steps,
            classifier_guidance=args.classifier_guidance,
            classifier_guidance_scale=args.classifier_guidance_scale,
            classifier_guidance_steps=args.classifier_guidance_steps,
            importance_sampling=args.importance_sampling,
            property_tau=args.property_tau,
            every_importance_t=args.every_importance_t,
            importance_sampling_start=args.importance_sampling_start,
            importance_sampling_end=args.importance_sampling_end,
            ckpt_property_model=args.ckpt_property_model,
            minimize_property=args.minimize_property,
            device="cpu",
            renormalize_property=args.renormalize_property,
        )
    else:
        print("\nStarting sampling...\n")
        results_dict, generated_smiles, stable_molecules = model.run_evaluation(
            step=0,
            dataset_info=model.dataset_info,
            ngraphs=args.ngraphs,
            bs=args.batch_size,
            return_molecules=True,
            verbose=True,
            inner_verbose=True,
            save_dir=args.save_dir,
            ddpm=not args.ddim,
            eta_ddim=args.eta_ddim,
            run_test_eval=True,
            save_traj=args.save_traj,
            fix_noise_and_nodes=args.fix_noise_and_nodes,
            n_nodes=args.n_nodes,
            vary_n_nodes=args.vary_n_nodes,
            relax_sampling=args.relax_sampling,
            relax_steps=args.relax_steps,
            classifier_guidance=args.classifier_guidance,
            classifier_guidance_scale=args.classifier_guidance_scale,
            classifier_guidance_steps=args.classifier_guidance_steps,
            importance_sampling=args.importance_sampling,
            property_tau=args.property_tau,
            every_importance_t=args.every_importance_t,
            importance_sampling_start=args.importance_sampling_start,
            importance_sampling_end=args.importance_sampling_end,
            ckpt_property_model=args.ckpt_property_model,
            minimize_property=args.minimize_property,
            device="cpu",
            renormalize_property=args.renormalize_property,
        )

    print("\nFinished sampling!\n")
    atom_decoder = stable_molecules[0].atom_decoder

    if args.calculate_energy:
        energies = []
        forces_norms = []
        print("Calculating energies...")
        for i in range(len(stable_molecules)):
            atom_types = [atom_decoder[int(a)] for a in stable_molecules[i].atom_types]
            try:
                e, f = calculate_xtb_energy(stable_molecules[i].positions, atom_types)
            except:
                continue
            stable_molecules[i].energy = e
            stable_molecules[i].forces_norm = f
            energies.append(e)
            forces_norms.append(f)

        print(f"Mean energies: {np.mean(energies)}")
        print(f"Mean force norms: {np.mean(forces_norms)}")

    if args.calculate_props:
        polarizabilities = []
        sm = stable_molecules.copy()
        stable_molecules = []
        print("Calculating properties...")
        for mol in tqdm(sm):
            atom_types = [atom_decoder[int(a)] for a in mol.atom_types]

            if prop_dist is not None and hparams.context_mapping:
                for j, key in enumerate(hparams.properties_list):
                    mean, mad = (
                        prop_dist.normalizer[key]["mean"],
                        prop_dist.normalizer[key]["mad"],
                    )
                    prop = mol.context[j] * mad + mean
                    mol.context = float(prop)
            try:
                charge = mol.charges.sum().item()
                results = xtb_calculate(
                    atoms=atom_types,
                    coords=mol.positions.tolist(),
                    charge=charge,
                    options={"grad": True},
                )
                for key, value in results.items():
                    mol.__setattr__(key, value)

                polarizabilities.append(mol.polarizability)
                stable_molecules.append(mol)
            except Exception as e:
                print(e)
                continue
        print(f"Mean polarizability: {np.mean(polarizabilities)}")

    if args.save_xyz:
        context = []
        for i in range(len(stable_molecules)):
            types = [atom_decoder[int(a)] for a in stable_molecules[i].atom_types]
            write_xyz_file(
                stable_molecules[i].positions,
                types,
                os.path.join(args.save_dir, f"mol_{i}.xyz"),
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

    if prop_dist is not None and args.save_xyz:
        with open(os.path.join(args.save_dir, "context.pickle"), "wb") as f:
            pickle.dump(context, f)
    if args.calculate_energy:
        with open(os.path.join(args.save_dir, "energies.pickle"), "wb") as f:
            pickle.dump(energies, f)
        with open(os.path.join(args.save_dir, "forces_norms.pickle"), "wb") as f:
            pickle.dump(forces_norms, f)
    with open(os.path.join(args.save_dir, "generated_smiles.pickle"), "wb") as f:
        pickle.dump(generated_smiles, f)
    with open(os.path.join(args.save_dir, "stable_molecules.pickle"), "wb") as f:
        pickle.dump(stable_molecules, f)
    with open(os.path.join(args.save_dir, "evaluation.pickle"), "wb") as f:
        pickle.dump(results_dict, f)
        
    if args.calculate_props:
        with open(os.path.join(args.save_dir, "polarizabilities.pickle"), "wb") as f:
            pickle.dump(polarizabilities, f)
            
    # save arguments
    argsdicts = vars(args)
    with open(os.path.join(args.save_dir, "args.json"), "w") as f:
        json.dump(argsdicts, f)

def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description='Data generation')
    parser.add_argument('--model-path', default="/hpfs/userws/cremej01/workspace/logs/aqm_qm7x/x0_t_weighting_dip_mpol/best_mol_stab.ckpt", type=str,
                        help='Path to trained model')
    parser.add_argument("--sample-only-valid", default=False, action="store_true")
    parser.add_argument('--save-dir', default="/hpfs/userws/cremej01/workspace/logs/aqm_qm7x/x0_t_weighting_dip_mpol", type=str,
                        help='Path to test output')
    parser.add_argument('--save-xyz', default=False, action="store_true",
                        help='Whether or not to store generated molecules in xyz files')
    parser.add_argument('--calculate-energy', default=False, action="store_true",
                        help='Whether or not to calculate xTB energies and forces')
    parser.add_argument('--calculate-props', default=False, action="store_true",
                        help='Whether or not to calculate xTB properties')
    parser.add_argument('--save-traj', default=False, action="store_true",
                        help='Whether or not to save whole trajectory')
    parser.add_argument('--relax-sampling', default=False, action="store_true",
                        help='Whether or not to relax using denoising with timestep 0')
    parser.add_argument('--relax-steps', default=10, type=int, help='How many denoising relaxation steps')
    parser.add_argument('--batch-size', default=80, type=int,
                            help='Batch-size to generate the selected ngraphs. Defaults to 80.')
    parser.add_argument('--ddim', default=False, action="store_true",
                        help='If DDIM sampling should be used. Defaults to False')
    parser.add_argument('--eta-ddim', default=1.0, type=float,
                        help='How to scale the std of noise in the reverse posterior. \
                            Can also be used for DDPM to track a deterministic trajectory. \
                            Defaults to 1.0')
    # number of nodes to sample
    parser.add_argument('--ngraphs', default=5000, type=int,
                            help='How many graphs to sample. Defaults to 5000')
    parser.add_argument('--fix-noise-and-nodes', default=False, action="store_true",
                        help='Whether or not to fix noise, e.g., for interpolation or guidance')
    parser.add_argument('--n-nodes', default=None, type=int,
                            help='Number of pre-defined nodes per molecule to sample')
    parser.add_argument('--vary-n-nodes', default=None, type=int,
                            help='Adds randomly up to specified nodes to pre-defined nodes per molecule to sample')
    # load external models
    parser.add_argument("--ckpt-property-model", default=None, type=str)
    # classifier-guidance
    parser.add_argument("--classifier-guidance", default=False, action="store_true")
    parser.add_argument('--classifier-guidance-scale', default=1.0e-4, type=float,
                        help='How to scale the guidance shift')
    parser.add_argument('--classifier-guidance-steps', default=100, type=int,
                        help='How many guidance steps')
    # importance sampling
    parser.add_argument("--importance-sampling", default=False, action="store_true")
    parser.add_argument("--property-tau", default=0.1, type=float)
    parser.add_argument("--every-importance-t", default=5, type=int)
    parser.add_argument("--importance-sampling-start", default=None, type=int)
    parser.add_argument("--importance-sampling-end", default=None, type=int)
    parser.add_argument("--minimize-property", default=False, action="store_true")
    parser.add_argument("--renormalize-property", default=False, action="store_true")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    # Evaluate negative log-likelihood for the test partitions
    evaluate(args)
