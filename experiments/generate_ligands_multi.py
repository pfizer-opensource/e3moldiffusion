import argparse
import os
import random
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
from experiments.data.utils import load_pickle, mol_to_torch_geometric, save_pickle
from experiments.docking import calculate_qvina2_score
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


def create_list_defaultdict():
    return defaultdict(list)


def evaluate(args):
    # load hyperparameter
    hparams = torch.load(args.model_path)["hyper_parameters"]
    hparams["select_train_subset"] = False
    hparams["diffusion_pretraining"] = False
    hparams["num_charge_classes"] = 6
    if args.dataset_root is not None:
        hparams["dataset_root"] = args.dataset_root
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
        args.model_path,
        dataset_info=dataset_info,
        smiles_list=train_smiles,
        histogram=histogram,
        prop_norm=prop_norm,
        prop_dist=prop_dist,
        load_ckpt_from_pretrained=None,
        ligand_pocket_interaction=False,
        load_ckpt=None,
        run_evaluation=True,
        strict=False,
    ).to(device)
    model = model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    args.save_dir.mkdir(exist_ok=args.skip_existing)
    raw_sdf_dir = Path(args.save_dir, "sampled")
    raw_sdf_dir.mkdir(exist_ok=args.skip_existing)
    if args.filter_by_docking_scores:
        docked_sdf_dir = Path(args.save_dir, "docked")
        docked_sdf_dir.mkdir(exist_ok=args.skip_existing)
    times_dir = Path(args.save_dir, "pocket_times")
    times_dir.mkdir(exist_ok=args.skip_existing)

    test_files = list(args.test_dir.glob("[!.]*.sdf"))
    if args.test_list is not None:
        with open(args.test_list, "r") as f:
            test_list = set(f.read().split(","))
        test_files = [x for x in test_files if x.stem in test_list]

    time_per_pocket = {}

    statistics_dict = defaultdict(list)
    buster_dict = defaultdict(list)
    violin_dict = defaultdict(list)
    posecheck_dict = defaultdict(list)

    sdf_files = []
    if args.encode_ligand:
        embedding_dict = defaultdict(create_list_defaultdict)
        embed_out_file = Path(args.save_dir, f"{args.mp_index}_latent_embeddings.pt")

    if args.build_obabel_mol:
        print(
            "Sampled molecules will be built with OpenBabel (without bond information)!"
        )
    print("\nStarting sampling...\n")

    assert np.sum([len(i) for i in split_list(test_files, args.num_gpus)]) == len(
        test_files
    )

    test_files = split_list(test_files, args.num_gpus)[args.mp_index - 1]

    print(f"Processing {len(test_files)} SDF files on job index {args.mp_index}.")

    for sdf_file in test_files:
        ligand_name = sdf_file.stem

        pdb_name, pocket_id, *suffix = ligand_name.split("_")
        pdb_file = Path(sdf_file.parent, f"{pdb_name}.pdb")
        txt_file = Path(sdf_file.parent, f"{ligand_name}.txt")
        sdf_out_file_raw = Path(raw_sdf_dir, f"{ligand_name}_gen.sdf")
        if args.filter_by_docking_scores:
            sdf_out_file_docked = Path(docked_sdf_dir, f"{ligand_name}_out.sdf")
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
            repeats=args.batch_size,
            device=device,
            ligand_sdf=sdf_file if not args.encode_ligand else None,
        )

        if args.encode_ligand:
            suppl = Chem.SDMolSupplier(str(sdf_file))
            mol = []
            for m in suppl:
                mol.append(m)
            assert len(mol) == 1
            mol = mol[0]
            ligand_data = mol_to_torch_geometric(
                mol,
                dataset_info.atom_encoder,
                smiles=Chem.MolToSmiles(mol),
                remove_hydrogens=hparams.remove_hs,
                cog_proj=True,  # only for processing the ligand-shape encode
            )
            ligand_data = Batch.from_data_list([ligand_data]).to(device)
            with torch.no_grad():
                ligand_embeds = model.encode_ligand(
                    pos=ligand_data.pos,
                    atom_types=ligand_data.x,
                    data_batch=torch.zeros_like(ligand_data.x).to(device),
                    bond_edge_index=ligand_data.edge_index,
                    bond_edge_attr=ligand_data.edge_attr,
                )
            embedding_dict[ligand_name]["seed"].append(ligand_embeds)

            ligand_data = Batch.from_data_list(
                [deepcopy(ligand_data) for _ in range(args.batch_size)]
            )
            for name, tensor in ligand_data.to_dict().items():
                pocket_data.__setattr__(name, tensor)

        start = datetime.now()

        k = 0
        while (
            len(valid_and_unique_molecules) < args.num_ligands_per_pocket
            and k <= args.max_sample_iter
        ):
            k += 1
            with torch.no_grad():
                molecules = model.generate_ligands(
                    pocket_data,
                    num_graphs=args.batch_size,
                    fix_n_nodes=args.fix_n_nodes,
                    vary_n_nodes=args.vary_n_nodes,
                    n_nodes_bias=args.n_nodes_bias,
                    property_guidance=args.property_guidance,
                    ckpt_property_model=args.ckpt_property_model,
                    property_self_guidance=args.property_self_guidance,
                    property_guidance_complex=args.property_guidance_complex,
                    guidance_scale=args.guidance_scale,
                    build_obabel_mol=args.build_obabel_mol,
                    inner_verbose=False,
                    save_traj=False,
                    ddpm=not args.ddim,
                    eta_ddim=args.eta_ddim,
                    relax_mol=args.relax_mol,
                    max_relax_iter=args.max_relax_iter,
                    sanitize=args.sanitize,
                    importance_sampling=args.importance_sampling,  # True
                    tau=args.tau,  # 0.1,
                    every_importance_t=args.every_importance_t,  # 5,
                    importance_sampling_start=args.importance_sampling_start,  # 0,
                    importance_sampling_end=args.importance_sampling_end,  # 200,
                    maximize_score=True,
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
                filter_by_posebusters=args.filter_by_posebusters,
                filter_by_lipinski=args.filter_by_lipinski,
                pdb_file=pdb_file,
                remove_hs=hparams.remove_hs,
                device="cpu",
            )

            valid_and_unique_molecules = valid_molecules.copy()
            tmp_molecules = valid_molecules.copy()

        if len(valid_and_unique_molecules) < args.num_ligands_per_pocket and (
            args.filter_by_posebusters or args.filter_by_lipinski
        ):
            k = 0
            while (
                len(valid_and_unique_molecules) < args.num_ligands_per_pocket
                and k <= args.max_sample_iter
            ):
                k += 1
                with torch.no_grad():
                    molecules = model.generate_ligands(
                        pocket_data,
                        num_graphs=args.batch_size,
                        fix_n_nodes=args.fix_n_nodes,
                        vary_n_nodes=args.vary_n_nodes,
                        n_nodes_bias=args.n_nodes_bias,
                        build_obabel_mol=args.build_obabel_mol,
                        inner_verbose=False,
                        save_traj=False,
                        ddpm=not args.ddim,
                        eta_ddim=args.eta_ddim,
                        relax_mol=args.relax_mol,
                        max_relax_iter=args.max_relax_iter,
                        sanitize=args.sanitize,
                        importance_sampling=args.importance_sampling,  # True
                        tau=args.tau,  # 0.1,
                        every_importance_t=args.every_importance_t,  # 5,
                        importance_sampling_start=args.importance_sampling_start,  # 0,
                        importance_sampling_end=args.importance_sampling_end,  # 200,
                        maximize_score=True,
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
                    pdb_file=pdb_file,
                    remove_hs=hparams.remove_hs,
                    device="cpu",
                )
                valid_and_unique_molecules = valid_molecules.copy()
                tmp_molecules = valid_molecules.copy()

        if len(valid_and_unique_molecules) == 0:
            print(
                f"Reached {args.max_sample_iter} sampling iterations, but could not find any ligands for pdb file {pdb_file}. Skipping."
            )
            continue
        elif len(valid_and_unique_molecules) < args.num_ligands_per_pocket:
            print(
                f"FYI: Reached {args.max_sample_iter} sampling iterations, but could only find {len(valid_and_unique_molecules)} ligands for pdb file {pdb_file}."
            )

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

        if args.filter_by_docking_scores:
            temp_file = tempfile.NamedTemporaryFile(
                suffix=f"{random.randint(0, 100000)}.sdf", delete=False
            )
            temp_path = temp_file.name
            write_sdf_file(temp_path, valid_molecules, extract_mol=True)
            target = ("-").join(
                ligand_name.split("-")[:5]
            )  # get the target, chain and ligand name
            ground_truth_score = load_pickle(args.docking_scores)[
                target
            ]  # get the ground truth docking score of that target
            receptor_file = Path(args.pdbqt_dir, ligand_name.split("_")[0] + ".pdbqt")
            scores, rdmols, valid_ids = calculate_qvina2_score(
                pdb_file,
                receptor_file,
                temp_path,
                args.save_dir,
                return_rdmol=True,
                filtering=True,
            )
            valid_molecules = [
                m for i, m in enumerate(valid_molecules) if i in valid_ids
            ]
            valid_molecules = [
                m for m, s in zip(valid_molecules, scores) if ground_truth_score < s
            ]
            write_sdf_file(sdf_out_file_docked, rdmols)

            temp_file.close()
            os.remove(temp_path)

            if len(valid_molecules) == 0:
                print("No sample found with better docking score. Skipping!")
                continue

        run_time = datetime.now() - start
        print(f"\n Run time={run_time} for {len(valid_molecules)} valid molecules \n")

        statistics_dict["QED"].append(statistics["QED"])
        statistics_dict["SA"].append(statistics["SA"])
        statistics_dict["Lipinski"].append(statistics["Lipinski"])
        statistics_dict["Diversity"].append(statistics["Diversity"])

        if len(valid_molecules) > args.num_ligands_per_pocket:
            valid_molecules = valid_molecules[
                : args.num_ligands_per_pocket
            ]  # we could sort them by QED, SA or whatever

        if args.encode_ligand:
            ligand_data = [
                mol_to_torch_geometric(
                    mol.rdkit_mol,
                    dataset_info.atom_encoder,
                    Chem.MolToSmiles(mol.rdkit_mol),
                    remove_hydrogens=True,
                    cog_proj=True,
                )
                for mol in valid_molecules
            ]
            ligand_data = Batch.from_data_list(ligand_data).to(device)
            with torch.no_grad():
                ligand_embeds = model.encode_ligand(
                    pos=ligand_data.pos,
                    atom_types=ligand_data.x,
                    data_batch=ligand_data.batch,
                    bond_edge_index=ligand_data.edge_index,
                    bond_edge_attr=ligand_data.edge_attr,
                )
            embedding_dict[ligand_name]["sampled"].append(ligand_embeds)

        write_sdf_file(sdf_out_file_raw, valid_molecules, extract_mol=True)
        sdf_files.append(sdf_out_file_raw)

        # PoseBusters
        if not args.filter_by_posebusters and not args.omit_posebusters:
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

        if not args.omit_posecheck:
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

    with open(Path(args.save_dir, "pocket_times.txt"), "w") as f:
        for k, v in time_per_pocket.items():
            f.write(f"{k} {v}\n")

    times_arr = torch.tensor([x for x in time_per_pocket.values()])
    print(
        f"Time per pocket: {times_arr.mean():.3f} \pm "
        f"{times_arr.std(unbiased=False):.2f}"
    )
    print("Sampling finished.")

    save_pickle(
        statistics_dict,
        os.path.join(args.save_dir, f"{args.mp_index}_statistics_dict.pickle"),
    )
    if not args.filter_by_posebusters and not args.omit_posebusters:
        save_pickle(
            buster_dict,
            os.path.join(args.save_dir, f"{args.mp_index}_posebusters_sampled.pickle"),
        )
    if not args.omit_posebusters and not args.omit_posecheck:
        save_pickle(
            violin_dict,
            os.path.join(args.save_dir, f"{args.mp_index}_violin_dict_sampled.pickle"),
        )
    if not args.omit_posecheck:
        save_pickle(
            posecheck_dict,
            os.path.join(args.save_dir, f"{args.mp_index}_posecheck_sampled.pickle"),
        )

    if args.encode_ligand:
        embedding_dict = {k: v for k, v in embedding_dict.items()}
        torch.save(embedding_dict, embed_out_file)


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
        "--pdbqt-dir",
        type=Path,
        default=None,
        help="Directory where all full protein pdbqt files are stored. If not available, there will be calculated on the fly.",
    )
    parser.add_argument("--test-list", type=Path, default=None)
    parser.add_argument("--save-dir", type=Path)
    parser.add_argument("--fix-n-nodes", action="store_true")
    parser.add_argument("--vary-n-nodes", action="store_true")
    parser.add_argument("--n-nodes-bias", default=0, type=int)
    parser.add_argument("--property-guidance", default=False, action="store_true")
    parser.add_argument("--ckpt-property-model", default=None, type=str)
    parser.add_argument("--property-self-guidance", default=False, action="store_true")
    parser.add_argument("--property-guidance-complex", default=False, action="store_true")
    parser.add_argument("--guidance-scale", default=1.e-4, type=float)
    parser.add_argument("--filter-by-posebusters", action="store_true")
    parser.add_argument("--filter-by-lipinski", action="store_true")
    parser.add_argument("--filter-by-docking-scores", action="store_true", 
                        help="Samples will be docked directly after generation and filtered versus a ground truth docking score. Only higher score will be kept.")
    parser.add_argument("--docking-scores", type=Path, default=None, 
                        help="If filter-by-docking-score is set to True, you have to provide ground-truth docking scores as a dictionary containing the respective ground truth ligand names and their scores")
    parser.add_argument("--omit-posebusters", default=False, action="store_true")
    parser.add_argument("--omit-posecheck", default=False, action="store_true")

    # importance sampling
    parser.add_argument("--importance-sampling", default=False, action="store_true")
    parser.add_argument("--tau", default=0.1, type=float)
    parser.add_argument("--every-importance-t", default=5, type=int)
    parser.add_argument("--importance-sampling-start", default=0, type=int)
    parser.add_argument("--importance-sampling-end", default=250, type=int)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    evaluate(args)
