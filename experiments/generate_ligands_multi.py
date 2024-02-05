import argparse
import os
import shutil
import tempfile
import warnings
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from time import time

import numpy as np
import torch
from Bio.PDB import PDBParser
from posebusters import PoseBusters
from posecheck.posecheck import PoseCheck
from rdkit import Chem
from torch_geometric.data import Batch

from experiments.data.distributions import DistributionProperty
from experiments.data.ligand.process_pdb import get_pdb_components, write_pdb
from experiments.data.utils import mol_to_torch_geometric, save_pickle
from experiments.sampling.analyze import analyze_stability_for_molecules
from experiments.utils import (
    prepare_pocket,
    retrieve_interactions_per_mol,
    split_list,
    write_sdf_file,
)

warnings.filterwarnings(
    "ignore", category=UserWarning, message="TypedStorage is deprecated"
)
warnings.filterwarnings("ignore", category=DeprecationWarning)


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def evaluate(
    mp_index,
    num_gpus,
    model_path,
    save_dir,
    test_dir,
    pdb_dir,
    test_list,
    skip_existing,
    fix_n_nodes,
    vary_n_nodes,
    n_nodes_bias,
    num_ligands_per_pocket,
    build_obabel_mol,
    batch_size,
    ddpm,
    eta_ddim,
    relax_mol,
    max_relax_iter,
    sanitize,
    filter_by_posebusters,
    max_sample_iter,
    dataset_root=None,
    encode_ligand=False,
    importance_sampling=False,
    ground_truth_size=False,
):
    # load hyperparameter
    hparams = torch.load(model_path)["hyper_parameters"]
    hparams["select_train_subset"] = False
    hparams["diffusion_pretraining"] = False
    hparams["num_charge_classes"] = 6
    if dataset_root is not None:
        hparams["dataset_root"] = dataset_root
    hparams = dotdict(hparams)

    hparams.load_ckpt_from_pretrained = None
    hparams.store_intermediate_coords = False
    hparams.load_ckpt = None
    hparams.gpus = 1

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

    datamodule = DataModule(hparams)

    from experiments.data.data_info import GeneralInfos as DataInfos

    dataset_info = DataInfos(datamodule, hparams)
    histogram = os.path.join(hparams.dataset_root, "size_distribution.npy")
    histogram = np.load(histogram).tolist()
    train_smiles = list(datamodule.train_dataset.smiles)

    prop_norm, prop_dist = None, None
    if len(hparams.properties_list) > 0 and hparams.context_mapping:
        prop_norm = datamodule.compute_mean_mad(hparams.properties_list)
        prop_dist = DistributionProperty(datamodule, hparams.properties_list)
        prop_dist.set_normalizer(prop_norm)

    if hparams.additional_feats:
        from experiments.diffusion_discrete_pocket_addfeats import Trainer

        # from experiments.diffusion_discrete_pocket_addfeats_reduced import Trainer
    elif hparams.latent_dim is None:
        from experiments.diffusion_discrete_pocket import Trainer
    else:
        from experiments.diffusion_discrete_latent_pocket_ligand import Trainer

    torch.cuda.empty_cache()

    # if you want bond_model_guidance, flag this here in the Trainer
    device = "cuda"
    model = Trainer.load_from_checkpoint(
        model_path,
        dataset_info=dataset_info,
        smiles_list=train_smiles,
        histogram=histogram,
        prop_norm=prop_norm,
        prop_dist=prop_dist,
        load_ckpt_from_pretrained=None,
        store_intermediate_coords=False,
        ligand_pocket_hidden_distance=None,
        ligand_pocket_distance_loss=None,
        load_ckpt=None,
        # energy_model_guidance=True if use_energy_guidance else False,
        # ckpt_energy_model=ckpt_energy_model,
        run_evaluation=True,
        strict=False,
    ).to(device)
    model = model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    save_dir.mkdir(exist_ok=skip_existing)
    raw_sdf_dir = Path(save_dir, "sampled")
    raw_sdf_dir.mkdir(exist_ok=skip_existing)
    times_dir = Path(save_dir, "pocket_times")
    times_dir.mkdir(exist_ok=skip_existing)

    test_files = list(test_dir.glob("[!.]*.sdf"))
    if test_list is not None:
        with open(test_list, "r") as f:
            test_list = set(f.read().split(","))
        test_files = [x for x in test_files if x.stem in test_list]

    time_per_pocket = {}

    statistics_dict = defaultdict(list)
    buster_dict = defaultdict(list)
    violin_dict = defaultdict(list)
    posecheck_dict = defaultdict(list)

    sdf_files = []

    if build_obabel_mol:
        print(
            "Sampled molecules will be built with OpenBabel (without bond information)!"
        )
    print("\nStarting sampling...\n")

    assert np.sum([len(i) for i in split_list(test_files, num_gpus)]) == len(test_files)

    test_files = split_list(test_files, num_gpus)[mp_index - 1]

    print(f"Processing {len(test_files)} SDF files on job index {mp_index}.")

    for sdf_file in test_files:
        ligand_name = sdf_file.stem

        pdb_name, pocket_id, *suffix = ligand_name.split("_")
        pdb_file = Path(sdf_file.parent, f"{pdb_name}.pdb")
        txt_file = Path(sdf_file.parent, f"{ligand_name}.txt")
        sdf_out_file_raw = Path(raw_sdf_dir, f"{ligand_name}_gen.sdf")
        time_file = Path(times_dir, f"{ligand_name}.txt")

        t_pocket_start = time()

        with open(txt_file, "r") as f:
            resi_list = f.read().split()

        pdb_struct = PDBParser(QUIET=True).get_structure("", pdb_file)[0]
        if resi_list is not None:
            # define pocket with list of residues
            residues = [
                pdb_struct[x.split(":")[0]][(" ", int(x.split(":")[1]), " ")]
                for x in resi_list
            ]

        all_molecules = 0
        tmp_molecules = []
        valid_and_unique_molecules = []

        pocket_data = prepare_pocket(
            residues,
            dataset_info.atom_encoder,
            no_H=True,
            repeats=batch_size,
            device=device,
        )
        
        suppl = Chem.SDMolSupplier(str(sdf_file))
        mol = []
        for m in suppl:
            mol.append(m)
        assert len(mol) == 1
        mol = mol[0]
        num_nodes_ligand = mol.GetNumAtoms()
        num_nodes_ligand = torch.tensor([num_nodes_ligand] * batch_size).long().to(device)
        
        pocket_data.num_nodes_ligand = num_nodes_ligand
        if encode_ligand:
            ligand_data = mol_to_torch_geometric(
                mol,
                dataset_info.atom_encoder,
                smiles=Chem.MolToSmiles(mol),
                remove_hydrogens=hparams.remove_hs,
                cog_proj=True,  # only for processing the ligand-shape encode
            )
            ligand_data = Batch.from_data_list(
                [deepcopy(ligand_data) for _ in range(batch_size)]
            )
            for name, tensor in ligand_data.to_dict().items():
                pocket_data.__setattr__(name, tensor)

        start = datetime.now()

        k = 0
        while (
            len(valid_and_unique_molecules) < num_ligands_per_pocket
            and k <= max_sample_iter
        ):
            k += 1
            with torch.no_grad():
                
                if not importance_sampling:
                    molecules = model.generate_ligands(
                        pocket_data,
                        num_graphs=batch_size,
                        fix_n_nodes=fix_n_nodes,
                        vary_n_nodes=vary_n_nodes,
                        n_nodes_bias=n_nodes_bias,
                        build_obabel_mol=build_obabel_mol,
                        inner_verbose=False,
                        save_traj=False,
                        ddpm=ddpm,
                        eta_ddim=eta_ddim,
                        relax_mol=relax_mol,
                        max_relax_iter=max_relax_iter,
                        sanitize=sanitize,
                        ground_truth_size=ground_truth_size,
                    )
                else:
                    molecules = model.generate_ligands(
                        pocket_data,
                        num_graphs=batch_size,
                        fix_n_nodes=fix_n_nodes,
                        vary_n_nodes=vary_n_nodes,
                        n_nodes_bias=n_nodes_bias,
                        build_obabel_mol=build_obabel_mol,
                        inner_verbose=False,
                        save_traj=False,
                        ddpm=ddpm,
                        eta_ddim=eta_ddim,
                        relax_mol=relax_mol,
                        max_relax_iter=max_relax_iter,
                        sanitize=sanitize,
                        importance_sampling=True,
                        tau=0.1,
                        every_importance_t=5,
                        importance_sampling_start=0,
                        importance_sampling_end=200,
                        maximize_score=True,
                        ground_truth_size=ground_truth_size,
                    )
                    
            all_molecules += len(molecules)
            tmp_molecules.extend(molecules)
            valid_molecules = analyze_stability_for_molecules(
                molecule_list=tmp_molecules,
                dataset_info=dataset_info,
                smiles_train=train_smiles,
                local_rank=0,
                return_molecules=True,
                calculate_statistics=False,
                calculate_distribution_statistics=False,
                filter_by_posebusters=filter_by_posebusters,
                pdb_file=pdb_file,
                remove_hs=hparams.remove_hs,
                device="cpu",
            )
            valid_and_unique_molecules = valid_molecules.copy()
            tmp_molecules = valid_molecules.copy()

        if (
            len(valid_and_unique_molecules) < num_ligands_per_pocket
            and filter_by_posebusters
        ):
            k = 0
            while (
                len(valid_and_unique_molecules) < num_ligands_per_pocket
                and k <= max_sample_iter
            ):
                k += 1
                with torch.no_grad():
                    if not importance_sampling:
                        molecules = model.generate_ligands(
                            pocket_data,
                            num_graphs=batch_size,
                            fix_n_nodes=fix_n_nodes,
                            vary_n_nodes=vary_n_nodes,
                            n_nodes_bias=n_nodes_bias,
                            build_obabel_mol=build_obabel_mol,
                            inner_verbose=False,
                            save_traj=False,
                            ddpm=ddpm,
                            eta_ddim=eta_ddim,
                            relax_mol=relax_mol,
                            max_relax_iter=max_relax_iter,
                            sanitize=sanitize,
                            ground_truth_size=ground_truth_size,
                        )
                    else:
                        molecules = model.generate_ligands(
                            pocket_data,
                            num_graphs=batch_size,
                            fix_n_nodes=fix_n_nodes,
                            vary_n_nodes=vary_n_nodes,
                            n_nodes_bias=n_nodes_bias,
                            build_obabel_mol=build_obabel_mol,
                            inner_verbose=False,
                            save_traj=False,
                            ddpm=ddpm,
                            eta_ddim=eta_ddim,
                            relax_mol=relax_mol,
                            max_relax_iter=max_relax_iter,
                            sanitize=sanitize,
                            importance_sampling=True,
                            tau=0.1,
                            every_importance_t=5,
                            importance_sampling_start=0,
                            importance_sampling_end=200,
                            maximize_score=True,
                            ground_truth_size=ground_truth_size,
                        )
                all_molecules += len(molecules)
                tmp_molecules.extend(molecules)
                valid_molecules = analyze_stability_for_molecules(
                    molecule_list=tmp_molecules,
                    dataset_info=dataset_info,
                    smiles_train=train_smiles,
                    local_rank=0,
                    return_molecules=True,
                    calculate_statistics=False,
                    calculate_distribution_statistics=False,
                    filter_by_posebusters=False,
                    pdb_file=pdb_file,
                    remove_hs=hparams.remove_hs,
                    device="cpu",
                )
                valid_and_unique_molecules = valid_molecules.copy()
                tmp_molecules = valid_molecules.copy()

        if len(valid_and_unique_molecules) < num_ligands_per_pocket:
            print(
                f"Reached {max_sample_iter} sampling iterations, but could not find {num_ligands_per_pocket} for pdb file {pdb_file}. Abort."
            )
            break

        del pocket_data
        torch.cuda.empty_cache()

        (
            _,
            validity_dict,
            statistics,
            _,
            _,
            valid_molecules,
        ) = analyze_stability_for_molecules(
            molecule_list=tmp_molecules,
            dataset_info=dataset_info,
            smiles_train=train_smiles,
            local_rank=0,
            return_molecules=True,
            calculate_statistics=True,
            calculate_distribution_statistics=False,
            return_stats_per_molecule=False,
            remove_hs=hparams.remove_hs,
            device="cpu",
        )
        run_time = datetime.now() - start
        print(f"\n Run time={run_time} for {len(valid_molecules)} valid molecules \n")

        statistics_dict["QED"].append(statistics["QED"])
        statistics_dict["SA"].append(statistics["SA"])
        statistics_dict["Lipinski"].append(statistics["Lipinski"])
        statistics_dict["Diversity"].append(statistics["Diversity"])

        valid_molecules = valid_molecules[
            :num_ligands_per_pocket
        ]  # we could sort them by QED, SA or whatever

        write_sdf_file(sdf_out_file_raw, valid_molecules, extract_mol=True)
        sdf_files.append(sdf_out_file_raw)

        # pdb_name = str(sdf_out_file_raw).split("/")[-1].split("-")[0]
        # if pdb_dir is not None:
        #     pdbs = [
        #         str(i).split("/")[-1].split(".")[0] for i in pdb_dir.glob("[!.]*.pdb")
        #     ]
        # else:
        #     pdbs = None
        # if pdbs is not None and pdb_name in pdbs:
        #     pdb_file = os.path.join(str(pdb_dir), pdb_name + ".pdb")
        # else:
        #     temp_dir = tempfile.mkdtemp()
        #     protein, _ = get_pdb_components(pdb_name)
        #     pdb_file = write_pdb(temp_dir, protein, pdb_name)

        # PoseBusters
        if not filter_by_posebusters:
            print("Starting evaluation with PoseBusters...")
            buster = {}
            buster_mol = PoseBusters(config="mol")
            buster_mol_df = buster_mol.bust([sdf_out_file_raw], None, None)
            for metric in buster_mol_df.columns:
                violin_dict[metric].extend(list(buster_mol_df[metric]))
                buster[metric] = buster_mol_df[metric].sum() / len(
                    buster_mol_df[metric]
                )
            buster_dock = PoseBusters(config="dock")
            buster_dock_df = buster_dock.bust([sdf_out_file_raw], None, str(pdb_file))
            for metric in buster_dock_df:
                if metric not in buster:
                    violin_dict[metric].extend(list(buster_dock_df[metric]))
                    buster[metric] = buster_dock_df[metric].sum() / len(
                        buster_dock_df[metric]
                    )
            for k, v in buster.items():
                buster_dict[k].append(v)
            print("Done!")

        # PoseCheck
        print("Starting evaluation with PoseCheck...")
        pc = PoseCheck()
        pc.load_protein_from_pdb(str(pdb_file))
        pc.load_ligands_from_sdf(str(sdf_out_file_raw), add_hs=True)
        interactions = pc.calculate_interactions()
        interactions_per_mol, interactions_mean = retrieve_interactions_per_mol(
            interactions
        )
        for k, v in interactions_per_mol.items():
            violin_dict[k].extend(v)
        for k, v in interactions_mean.items():
            posecheck_dict[k].append(v["mean"])
        clashes = pc.calculate_clashes()
        strain_energies = pc.calculate_strain_energy()
        violin_dict["Clashes"].extend(clashes)
        violin_dict["Strain Energies"].extend(strain_energies)
        posecheck_dict["Clashes"].append(np.mean(clashes))
        posecheck_dict["Strain Energies"].append(np.nanmedian(strain_energies))
        print("Done!")

        # try:
        #     shutil.rmtree(temp_dir)
        # except UnboundLocalError:
        #     pass

        # Time the sampling process
        time_per_pocket[str(sdf_file)] = time() - t_pocket_start
        with open(time_file, "w") as f:
            f.write(f"{str(sdf_file)} {time_per_pocket[str(sdf_file)]}")

    with open(Path(save_dir, "pocket_times.txt"), "w") as f:
        for k, v in time_per_pocket.items():
            f.write(f"{k} {v}\n")

    times_arr = torch.tensor([x for x in time_per_pocket.values()])
    print(
        f"Time per pocket: {times_arr.mean():.3f} \pm "
        f"{times_arr.std(unbiased=False):.2f}"
    )
    print("Sampling finished.")

    save_pickle(
        statistics_dict, os.path.join(save_dir, f"{mp_index}_statistics_dict.pickle")
    )
    if not filter_by_posebusters:
        save_pickle(
            buster_dict,
            os.path.join(save_dir, f"{mp_index}_posebusters_sampled.pickle"),
        )
    save_pickle(
        violin_dict, os.path.join(save_dir, f"{mp_index}_violin_dict_sampled.pickle")
    )
    save_pickle(
        posecheck_dict, os.path.join(save_dir, f"{mp_index}_posecheck_sampled.pickle")
    )


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description='Data generation')
    parser.add_argument('--mp-index', default=0, type=int)
    parser.add_argument("--num-gpus", default=8, type=int)
    parser.add_argument('--model-path', default="/hpfs/userws/cremej01/workspace/logs/aqm_qm7x/x0_t_weighting_dip_mpol/best_mol_stab.ckpt", type=str,
                        help='Path to trained model')
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument('--dataset-root', default=None, type=str,
                        help='If not set it will be taken from the model ckpt, otherwise it will overwrite it in the ckpt.')
    parser.add_argument("--use-energy-guidance", default=False, action="store_true")
    parser.add_argument("--ckpt-energy-model", default=None, type=str)
    parser.add_argument('--guidance-scale', default=1.0e-4, type=float,
                        help='How to scale the guidance shift')
    parser.add_argument('--save-xyz', default=False, action="store_true",
                        help='Whether or not to store generated molecules in xyz files')
    parser.add_argument('--calculate-energy', default=False, action="store_true",
                        help='Whether or not to calculate xTB energies and forces')
    parser.add_argument('--num-ligands-per-pocket', default=100, type=int,
                            help='How many ligands per pocket to sample. Defaults to 10')
    parser.add_argument("--build-obabel-mol", default=False, action="store_true")
    parser.add_argument('--batch-size', default=100, type=int)
    parser.add_argument('--ddim', default=False, action="store_true",
                        help='If DDIM sampling should be used. Defaults to False')
    parser.add_argument('--eta-ddim', default=1.0, type=float,
                        help='How to scale the std of noise in the reverse posterior. \
                            Can also be used for DDPM to track a deterministic trajectory. \
                            Defaults to 1.0')
    parser.add_argument("--relax-mol", default=False, action="store_true")
    parser.add_argument("--sanitize", default=False, action="store_true")
    parser.add_argument('--max-relax-iter', default=200, type=int,
                            help='How many iteration steps for UFF optimization')
    parser.add_argument('--max-sample-iter', default=20, type=int,
                            help='How many iteration steps for UFF optimization')
    parser.add_argument("--test-dir", type=Path)
    parser.add_argument("--encode-ligand", default=False, action="store_true")
    parser.add_argument(
        "--pdb-dir",
        type=Path,
        default=None,
        help="Directory where all full protein pdb files are stored. If not available, there will be calculated on the fly.",
    )
    parser.add_argument("--test-list", type=Path, default=None)
    parser.add_argument("--save-dir", type=Path)
    parser.add_argument("--fix-n-nodes", default=False, action="store_true")
    parser.add_argument("--vary-n-nodes", default=False,  action="store_true")
    parser.add_argument("--n-nodes-bias", default=0, type=int)
    parser.add_argument("--ground-truth-size", default=False, action="store_true")
    parser.add_argument("--filter-by-posebusters", default=False, action="store_true")
    parser.add_argument("--importance-sampling", action="store_true", default=False)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    # Evaluate negative log-likelihood for the test partitions
    evaluate(
        mp_index=args.mp_index,
        num_gpus=args.num_gpus,
        model_path=args.model_path,
        test_dir=args.test_dir,
        pdb_dir=args.pdb_dir,
        test_list=args.test_list,
        skip_existing=args.skip_existing,
        save_dir=args.save_dir,
        fix_n_nodes=args.fix_n_nodes,
        vary_n_nodes=args.vary_n_nodes,
        n_nodes_bias=args.n_nodes_bias,
        batch_size=args.batch_size,
        num_ligands_per_pocket=args.num_ligands_per_pocket,
        build_obabel_mol=args.build_obabel_mol,
        ddpm=not args.ddim,
        eta_ddim=args.eta_ddim,
        relax_mol=args.relax_mol,
        max_relax_iter=args.max_relax_iter,
        sanitize=args.sanitize,
        dataset_root=args.dataset_root,
        encode_ligand=args.encode_ligand,
        filter_by_posebusters=args.filter_by_posebusters,
        max_sample_iter=args.max_sample_iter,
        importance_sampling=args.importance_sampling,
        ground_truth_size=args.ground_truth_size,
    )
