import argparse
import os
import pickle
import warnings

import torch
from rdkit import Chem
from tqdm import tqdm

from experiments.data.distributions import DistributionProperty
from experiments.data.utils import write_xyz_file
from experiments.xtb_wrapper import xtb_calculate

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
    calculate_props=False,
    use_energy_guidance=False,
    ckpt_energy_model=None,
    guidance_scale=1.0e-4,
    ngraphs=5000,
    batch_size=80,
    step=0,
    ddpm=True,
    eta_ddim=1.0,
    every_k_step=1,
    max_num_batches=-1,
    guidance_start=None,
    calculate_relax_change=True,
    scaffold_elaboration=False,
    scaffold_hopping=False,
    resample_steps=1,
    fraction_new_nodes=0.0,
    T=500,
    relax_sampling=True,
    relax_steps=10,
    num_nodes=None,
    fix_noise_and_nodes=False,
    fixed_context=None,
    generate_ngraphs_valid=False,
):
    # load hyperparameter
    hparams = torch.load(model_path)["hyper_parameters"]
    hparams["select_train_subset"] = False
    hparams["diffusion_pretraining"] = False
    hparams = dotdict(hparams)
    hparams.num_charge_classes = 6

    hparams.dataset_root = "/scratch1/cremej01/data/aqm"

    hparams.load_ckpt_from_pretrained = None
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
            from experiments.data.geom.geom_dataset_adaptive_qm import (
                GeomQMDataModule as DataModule,
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
        if hparams.additional_feats and hparams.use_qm_props:
            print("Using additional RDKit and QM features")
            from experiments.diffusion_discrete_addfeats_qm import Trainer
        elif hparams.use_qm_props and not hparams.additional_feats:
            from experiments.diffusion_discrete_qm import Trainer
        elif hparams.additional_feats and not hparams.use_qm_props:
            from experiments.diffusion_discrete_addfeats import Trainer
        else:
            from experiments.diffusion_discrete import Trainer

    # if you want bond_model_guidance, flag this here in the Trainer
    device = "cuda"
    model = Trainer.load_from_checkpoint(
        model_path,
        dataset_info=dataset_info,
        smiles_list=train_smiles,
        prop_norm=prop_norm,
        prop_dist=prop_dist,
        load_ckpt_from_pretrained=None,
        # energy_model_guidance=True if use_energy_guidance else False,
        # ckpt_energy_model=ckpt_energy_model,
        run_evaluation=True,
        strict=False,
    ).to(device)
    model = model.eval()
    if scaffold_elaboration or scaffold_hopping:
        # somehow i can access the datamodule from model.trainer.datamodule
        # might be due to:
        # The loaded checkpoint was produced with Lightning v2.0.6, which is newer than your current Lightning version: v2.0.4
        model.datamodule = datamodule

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if use_energy_guidance and ckpt_energy_model is not None:
        print("Sampling with energy guidance!")

    if scaffold_elaboration or scaffold_hopping:
        assert (
            scaffold_elaboration != scaffold_hopping
        ), "Either scaffold elaboration or scaffold hopping can be used at a time"
        (
            results_dict,
            generated_smiles,
            stable_molecules,
        ) = model.run_fixed_substructure_evaluation(
            dataset_info=model.dataset_info,
            save_dir=save_dir,
            return_molecules=True,
            verbose=True,
            save_traj=True,
            inner_verbose=True,
            ddpm=ddpm,
            eta_ddim=eta_ddim,
            every_k_step=every_k_step,
            run_test_eval=True,
            guidance_scale=guidance_scale,
            use_energy_guidance=use_energy_guidance,
            ckpt_energy_model=ckpt_energy_model,
            use_scaffold_dataset_sizes=True,
            scaffold_elaboration=scaffold_elaboration,
            scaffold_hopping=scaffold_hopping,
            max_num_batches=max_num_batches,
            fraction_new_nodes=fraction_new_nodes,
            resample_steps=resample_steps,
            # T=T,
            device="cpu",
            relax_sampling=relax_sampling,
            relax_steps=relax_steps,
        )
    elif generate_ngraphs_valid:
        results_dict, generated_smiles, stable_molecules = model.generate_valid_samples(
            dataset_info=model.dataset_info,
            ngraphs=ngraphs,
            bs=batch_size,
            return_molecules=True,
            verbose=True,
            inner_verbose=True,
            save_dir=save_dir,
            ddpm=ddpm,
            eta_ddim=eta_ddim,
            guidance_scale=guidance_scale,
            ckpt_energy_model=ckpt_energy_model,
            use_energy_guidance=use_energy_guidance,
            fix_noise_and_nodes=fix_noise_and_nodes,
            num_nodes=num_nodes,
            fixed_context=fixed_context,
            device="cpu",
        )

    else:
        results_dict, generated_smiles, stable_molecules = model.run_evaluation(
            step=step,
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
            guidance_scale=guidance_scale,
            ckpt_energy_model=ckpt_energy_model,
            use_energy_guidance=use_energy_guidance,
            fix_noise_and_nodes=fix_noise_and_nodes,
            num_nodes=num_nodes,
            fixed_context=fixed_context,
            device="cpu",
        )

    atom_decoder = stable_molecules[0].dataset_info.atom_decoder

    if calculate_props:
        sm = stable_molecules.copy()
        stable_molecules = []
        print("Calculating properties...")
        for mol in tqdm(sm):
            atom_types = [atom_decoder[int(a)] for a in mol.atom_types]

            for j, key in enumerate(hparams.properties_list):
                mean, mad = (
                    prop_dist.normalizer[key]["mean"],
                    prop_dist.normalizer[key]["mad"],
                )
                prop = mol.context[j] * mad + mean
                if len(hparams.properties_list) == 1:
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

                stable_molecules.append(mol)

            except Exception as e:
                print(e)
                continue

        # if hparams.dataset == "geomqm":
        #     import numpy as np

        #     # load precalculated target distribution
        #     target = datamodule.statistics["train"].force_norms
        #     generated_forces = []
        #     for m in stable_molecules:
        #         if m.normal_termination:
        #             generated_forces.extend(list(np.linalg.norm(m.grad, axis=1)))
        #     generated_forces = torch.tensor(generated_forces, dtype=torch.float)
        #     # calculate histograms with bin width 1e-5
        #     bin_width = 1e-5
        #     bins = torch.arange(0, 0.2, bin_width)
        #     generated, _ = torch.histogram(generated_forces, bins=bins)
        #     # calculate W1
        #     cs_generated = torch.cumsum(generated, dim=0)
        #     cs_target = torch.cumsum(target, dim=0)
        #     cs_generated /= cs_generated[-1].item()
        #     cs_target /= cs_target[-1].item()

        #     force_norm_w1 = (
        #         torch.sum(torch.abs(cs_generated - cs_target)).item() * bin_width
        #     )
        #     results_dict["ForceNormW1"] = force_norm_w1

    for key, value in results_dict.items():
        print(f"{key}:\t\t{value.item()}")

    if save_xyz:
        context = []
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
                    if len(hparams.properties_list) > 1:
                        prop = stable_molecules[i].context[j] * mad + mean
                    else:
                        prop = stable_molecules[i].context * mad + mean
                    tmp.append(float(prop))
                context.append(tmp)

    if prop_dist is not None and save_xyz:
        with open(os.path.join(save_dir, "context.pickle"), "wb") as f:
            pickle.dump(context, f)
    with open(os.path.join(save_dir, "generated_smiles.pickle"), "wb") as f:
        pickle.dump(generated_smiles, f)
    with open(os.path.join(save_dir, "stable_molecules.pickle"), "wb") as f:
        pickle.dump(stable_molecules, f)

    with open(os.path.join(save_dir, "results_dict.pickle"), "wb") as f:
        pickle.dump(results_dict, f)

    if calculate_relax_change:
        N_CORES = 8
        SAVE_XYZ_FILES = False

        import matplotlib.pyplot as plt
        import numpy as np

        from experiments.evaluate_geom_change import calc_diff_mols

        print(f"Calculating change of internal coordinates...")

        mols = [d.rdkit_mol for d in stable_molecules]
        for i, mol in enumerate(mols):
            try:
                Chem.SanitizeMol(mol)
            except Exception as e:
                print(f"Skipping molecule {i} because of sanitization error: {e}")

        diff_b_lengths, diff_b_angles, diff_d_angles, rmsds, diff_es = calc_diff_mols(
            mols, N_CORES, SAVE_XYZ_FILES
        )

        results_dict["delta_bond_lenghts"] = np.mean(np.abs(diff_b_lengths))
        results_dict["delta_bond_angles"] = np.mean(np.abs(diff_b_angles))
        tmp = np.abs(diff_d_angles)
        tmp[tmp >= 180.0] -= 180.0
        results_dict["delta_dihedrals"] = np.mean(tmp)
        results_dict["rmsds"] = np.mean(rmsds)
        results_dict["delta_energies"] = np.mean(np.abs(diff_es))

        for key, value in results_dict.items():
            print(f"{key}:\t\t{value.item()}")

        with open(os.path.join(save_dir, "results_dict.pickle"), "wb") as f:
            pickle.dump(results_dict, f)

        np.savetxt(os.path.join(save_dir, "eval_diff_b_lengths"), diff_b_lengths)
        np.savetxt(os.path.join(save_dir, "eval_diff_b_angles"), diff_b_angles)
        np.savetxt(os.path.join(save_dir, "eval_diff_d_angles"), diff_d_angles)
        np.savetxt(os.path.join(save_dir, "eval_rmsds"), rmsds)
        np.savetxt(os.path.join(save_dir, "eval_diff_energies"), diff_es)

        fig, ax = plt.subplots()
        ax.hist(diff_b_lengths, rwidth=0.9, bins=25)
        ax.set_xlabel("Δ Bond Length [Å]")
        ax.set_ylabel("Count")
        fig.savefig(os.path.join(save_dir, "diff_b_lengths.png"))

        fig, ax = plt.subplots()
        ax.hist(diff_b_angles, rwidth=0.9, bins=25)
        ax.set_xlabel("Δ Bond Angles [°]")
        ax.set_ylabel("Count")
        fig.savefig(os.path.join(save_dir, "diff_b_angles.png"))

        fig, ax = plt.subplots()
        diff_d_angles = np.abs(diff_d_angles)
        diff_d_angles[diff_d_angles >= 180.0] -= 180.0
        ax.hist(diff_d_angles, rwidth=0.9, bins=25)
        ax.set_xlabel("Δ Dihedral Angles [°]")
        ax.set_ylabel("Count")
        fig.savefig(os.path.join(save_dir, "diff_d_angles.png"))

        fig, ax = plt.subplots()
        ax.hist(rmsds, rwidth=0.9, bins=25)
        ax.set_xlabel("RMSD [Å]")
        ax.set_ylabel("Count")
        fig.savefig(os.path.join(save_dir, "rmsds.png"))

        fig, ax = plt.subplots()
        ax.hist(np.array(diff_es) * 627.509, rwidth=0.9, bins=25)
        ax.set_xlabel("Δ Energy [kcal/mol]")
        ax.set_ylabel("Count")
        fig.savefig(os.path.join(save_dir, "energies.png"))


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description='Data generation')
    parser.add_argument('--model-path', default="/scratch1/e3moldiffusion/logs/geom_qm/x0_snr_qm/best_mol_stab.ckpt", type=str,
                        help='Path to trained model')
    parser.add_argument("--use-energy-guidance", default=False, action="store_true")
    parser.add_argument("--calculate-relax-change", default=False, action="store_true")
    parser.add_argument("--ckpt-energy-model", default=None, type=str)
    parser.add_argument("--guidance-start", default=None, type=int)
    parser.add_argument('--guidance-scale', default=1.0e-4, type=float,
                        help='How to scale the guidance shift')
    parser.add_argument('--save-dir', default="/sharedhome/seumej/results", type=str,
                        help='Path to test output')
    parser.add_argument('--save-xyz', default=False, action="store_true",
                        help='Whether or not to store generated molecules in xyz files')
    parser.add_argument('--calculate-props', default=False, action="store_true",
                        help='Whether or not to calculate xTB properties')
    parser.add_argument('--generate-ngraphs-valid', default=False, action="store_true",
                        help='Sample as long as N valid molecules are sampled')
    parser.add_argument('--ngraphs', default=70, type=int,
                            help='How many graphs to sample. Defaults to 5000')
    parser.add_argument('--batch-size', default=70, type=int,
                            help='Batch-size to generate the selected ngraphs. Defaults to 80.')
    parser.add_argument('--ddim', default=False, action="store_true",
                        help='If DDIM sampling should be used. Defaults to False')
    parser.add_argument('--eta-ddim', default=1.0, type=float,
                        help='How to scale the std of noise in the reverse posterior. \
                            Can also be used for DDPM to track a deterministic trajectory. \
                            Defaults to 1.0')
    parser.add_argument('--fix-noise-and-nodes', default=False, action="store_true",
                        help='Whether or not to fix noise, e.g., for interpolation or guidance')
    parser.add_argument('--num-nodes', default=None, type=int,
                            help='Number of pre-defined nodes per molecule to sample')
    parser.add_argument('--fixed-context', default=None, nargs="+", type=float,
                            help='List of fixed property values')
    parser.add_argument('--scaffold-elaboration', default=False, action="store_true", help="Run scaffold elaboration")
    parser.add_argument('--scaffold-hopping', default=False, action="store_true", help="Run scaffold hopping")
    parser.add_argument('--resample-steps', default=1, type=int, help="Number of resampling steps for scaffold elaboration/hopping")
    parser.add_argument('--every-k-step', default=1, type=int, help="Jump k steps in denoising")
    parser.add_argument('--fraction-new-nodes', default=0.1, type=float, help="Fraction of new nodes to be added in scaffold elaboration/hopping")
    parser.add_argument('--max-num-batches', default=1, type=int, help="Maximum number of batches to use for scaffold elaboration/hopping")
    parser.add_argument('-T', default=500, type=int, help="T for denoising diffusion")
    parser.add_argument('--relax-sampling', default=False, action="store_true", help="Relax molecules")
    parser.add_argument('--relax-steps', default=10, type=int, help="Number of relaxation steps")
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
        every_k_step=args.every_k_step,
        max_num_batches=args.max_num_batches,
        save_xyz=args.save_xyz,
        calculate_props=args.calculate_props,
        use_energy_guidance=args.use_energy_guidance,
        ckpt_energy_model=args.ckpt_energy_model,
        guidance_scale=args.guidance_scale,
        guidance_start=args.guidance_start,
        calculate_relax_change=args.calculate_relax_change,
        scaffold_elaboration=args.scaffold_elaboration,
        scaffold_hopping=args.scaffold_hopping,
        resample_steps=args.resample_steps,
        fraction_new_nodes=args.fraction_new_nodes,
        T=args.T,
        relax_sampling=args.relax_sampling,
        relax_steps=args.relax_steps,
        fix_noise_and_nodes=args.fix_noise_and_nodes,
        num_nodes=args.num_nodes,
        fixed_context=args.fixed_context,
        generate_ngraphs_valid=args.generate_ngraphs_valid,
    )
