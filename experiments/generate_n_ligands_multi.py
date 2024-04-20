import argparse
import json
import os
import warnings
from collections import defaultdict
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
from experiments.data.utils import mol_to_torch_geometric, save_pickle
from experiments.sampling.analyze import analyze_stability_for_molecules
from experiments.utils import (
    chunks,
    prepare_data,
    prepare_data_and_generate_ligands,
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

    device = "cuda"
    model = Trainer.load_from_checkpoint(
        args.model_path,
        dataset_info=dataset_info,
        smiles_list=test_smiles,
        histogram=histogram,
        prop_norm=prop_norm,
        prop_dist=prop_dist,
        load_ckpt_from_pretrained=None,
        use_centroid_context_embed=hparams.use_centroid_context_embed,
        use_latent_encoder=hparams.use_latent_encoder,
        use_scaffold_latent_embed=hparams.use_scaffold_latent_embed,
        flow_matching=hparams.flow_matching,
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
    embedding_dict = None
    if args.encode_ligands:
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
        if args.test_dir_10A is not None:
            pdb_file_10A = Path(args.test_dir_10A, f"{pdb_name}.pdb")
            txt_file_10A = Path(args.test_dir_10A, f"{ligand_name}.txt")
            with open(txt_file_10A, "r") as f:
                resi_list_10A = f.read().split()
            pdb_struct_10A = PDBParser(QUIET=True).get_structure("", pdb_file_10A)[0]
            residues_10A = [
                pdb_struct_10A[x.split(":")[0]][(" ", int(x.split(":")[1]), " ")]
                for x in resi_list_10A
            ]
        else:
            residues_10A = None
        sdf_out_file_raw = Path(raw_sdf_dir, f"{ligand_name}_gen.sdf")
        time_file = Path(times_dir, f"{ligand_name}.txt")

        t_pocket_start = time()

        with open(txt_file, "r") as f:
            resi_list = f.read().split()

        assert (
            resi_list is not None
        ), "Pre-process pdb files extracting residues in specified cutoff and save as txt!"

        pdb_struct = PDBParser(QUIET=True).get_structure("", pdb_file)[0]

        residues = [
            pdb_struct[x.split(":")[0]][(" ", int(x.split(":")[1]), " ")]
            for x in resi_list
        ]

        all_molecules = 0
        molecules_list = []
        start = datetime.now()

        k = 0
        while (
            len(molecules_list) < args.num_ligands_per_pocket_to_sample
            and k <= args.max_sample_iter
        ):
            k += 1
            molecules = prepare_data_and_generate_ligands(
                model,
                residues,
                sdf_file,
                dataset_info,
                hparams=hparams,
                args=args,
                device=device,
                embedding_dict=embedding_dict,
                residues_10A=residues_10A,
            )

            all_molecules += len(molecules)
            molecules_list.extend(molecules)

        if len(molecules_list) == 0:
            print(
                f"Reached {args.max_sample_iter} sampling iterations, but could not find any ligands for pdb file {pdb_file}. Skipping."
            )
            continue
        elif len(molecules_list) < args.num_ligands_per_pocket_to_sample:
            print(
                f"FYI: Reached {args.max_sample_iter} sampling iterations, but could only find {len(molecules_list)} ligands for pdb file {pdb_file}."
            )

        run_time = datetime.now() - start
        print(f"\n Run time={run_time} for {len(molecules_list)} valid molecules \n")

        torch.cuda.empty_cache()

        if len(molecules_list) > args.num_ligands_per_pocket_to_save:
            indices = [i for i in range(args.num_ligands_per_pocket_to_save)]
            molecules_list = molecules_list[: args.num_ligands_per_pocket_to_save]
        else:
            indices = [i for i in range(len(molecules_list))]

        (
            _,
            validity_dict,
            statistics,
            _,
            _,
            molecules_list,
        ) = analyze_stability_for_molecules(
            molecule_list=molecules_list,
            dataset_info=dataset_info,
            smiles_train=test_smiles,
            local_rank=0,
            calculate_statistics=True,
            calculate_distribution_statistics=True,
            return_mean_stats=False,
            return_molecules=True,
            return_stats_per_molecule=True,
            return_valid=False,
            remove_hs=hparams.remove_hs,
            test=True,
            device="cpu",
        )

        for k, v in statistics.items():
            if isinstance(v, list):
                if len(v) >= len(indices):
                    v = [p for i, p in enumerate(v) if i in indices]
                    violin_dict[k].extend(v)
                    statistics_dict[k + "_mean"].append(np.mean(v))
                elif len(v) == 1:
                    statistics_dict[k].append(v[0])
            else:
                statistics_dict[k].append(v)

        statistics_dict["validity"].append(validity_dict["validity"])
        statistics_dict["uniqueness"].append(validity_dict["uniqueness"])
        statistics_dict["novelty"].append(validity_dict["novelty"])

        if args.encode_ligands:
            ligand_data = [
                mol_to_torch_geometric(
                    mol.rdkit_mol,
                    dataset_info.atom_encoder,
                    Chem.MolToSmiles(mol.rdkit_mol),
                    remove_hydrogens=True,
                    cog_proj=True,
                )
                for mol in molecules_list
            ]
            ligand_data = Batch.from_data_list(ligand_data).to(device)

            with torch.no_grad():
                ligand_embeds = model.encode_ligand(ligand_data)
            embedding_dict[ligand_name]["sampled"].append(ligand_embeds)
        if "ic50" in hparams.regression_property:
            # split into n chunks to avoid OOM error
            n = 4 if len(molecules_list) > 80 else 3
            molecules_list = list(chunks(molecules_list, n))
            ic50s = []
            for molecules in molecules_list:
                ligand_data = [
                    mol_to_torch_geometric(
                        mol.rdkit_mol,
                        dataset_info.atom_encoder,
                        Chem.MolToSmiles(mol.rdkit_mol),
                        remove_hydrogens=True,
                        cog_proj=False,
                    )
                    for mol in molecules
                ]
                ligand_data = Batch.from_data_list(ligand_data).to(device)
                pocket_data = prepare_data(
                    residues,
                    sdf_file,
                    dataset_info,
                    hparams,
                    args,
                    device,
                    batch_size=len(ligand_data.batch.bincount()),
                )
                pocket_data.update(ligand_data)

                t = torch.zeros((len(ligand_data),)).to(device).long()
                with torch.no_grad():
                    pred = model(pocket_data, t=t)
                ic50s.extend(pred["property_pred"][1].squeeze().detach().tolist())
            violin_dict["pIC50"].extend(ic50s)
            statistics_dict["pIC50_mean"].append(np.mean(ic50s))
            for i, ic50 in enumerate(ic50s):
                molecules_list[i].rdkit_mol.SetProp("pIC50", str(ic50))

        write_sdf_file(sdf_out_file_raw, molecules_list, extract_mol=True)
        sdf_files.append(sdf_out_file_raw)

        # PoseBusters
        if not args.filter_by_posebusters and not args.omit_posebusters:
            print("Starting evaluation with PoseBusters...")
            buster = {}
            buster_validity = 0.0
            buster_mol = PoseBusters(config="mol")
            buster_mol_df = buster_mol.bust([sdf_out_file_raw], None, None)
            for metric in buster_mol_df.columns:
                violin_dict[metric].extend(list(buster_mol_df[metric]))
                value = buster_mol_df[metric].sum() / len(buster_mol_df[metric])
                buster[metric] = value
                buster_validity += value
            buster_dock = PoseBusters(config="dock")
            buster_dock_df = buster_dock.bust([sdf_out_file_raw], None, str(pdb_file))
            for metric in buster_dock_df:
                if metric not in buster:
                    violin_dict[metric].extend(list(buster_dock_df[metric]))
                    value = buster_dock_df[metric].sum() / len(buster_dock_df[metric])
                    buster[metric] = value
                    buster_validity += value
            buster_validity /= len(buster)
            for k, v in buster.items():
                buster_dict[k].append(v)
            statistics_dict["pb_validity"].append(buster_validity)
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
    save_pickle(
        violin_dict,
        os.path.join(args.save_dir, f"{args.mp_index}_violin_dict_sampled.pickle"),
    )
    if not args.filter_by_posebusters and not args.omit_posebusters:
        save_pickle(
            buster_dict,
            os.path.join(args.save_dir, f"{args.mp_index}_posebusters_sampled.pickle"),
        )
    if not args.omit_posecheck:
        save_pickle(
            posecheck_dict,
            os.path.join(args.save_dir, f"{args.mp_index}_posecheck_sampled.pickle"),
        )

    if args.encode_ligands:
        embedding_dict = {k: v for k, v in embedding_dict.items()}
        torch.save(embedding_dict, embed_out_file)

    # save arguments
    argsdicts = vars(args)
    argsdicts = {str(k): str(v) for k, v in argsdicts.items()}
    savedirjson = os.path.join(str(args.save_dir), "args.json")
    with open(savedirjson, "w") as f:
        json.dump(argsdicts, f)


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
    parser.add_argument('--num-ligands-per-pocket-to-sample', default=100, type=int,
                            help='How many ligands per pocket to sample. Should be higher than num-ligands-per-pocket-to-save if filters, like sascore filtering, are active. Defaults to 100')
    parser.add_argument('--num-ligands-per-pocket-to-save', default=100, type=int,
                            help='How many ligands per pocket to save. Must be <= num-ligands-per-pocket-to-sample. Defaults to 100')
    parser.add_argument("--build-obabel-mol", default=False, action="store_true")
    parser.add_argument('--batch-size', default=100, type=int)
    parser.add_argument('--dist-cutoff', default=5.0, type=float)
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
    parser.add_argument("--test-dir-10A", type=Path, help="if specified, model takes 10A pocket config as ligand size prior.")
    parser.add_argument("--encode-ligands", default=False, action="store_true")
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
    parser.add_argument("--prior-n-atoms", default="targetdiff", type=str, choices=["conditional", "targetdiff"])

    parser.add_argument("--filter-by-posebusters", action="store_true")
    parser.add_argument("--filter-by-lipinski", action="store_true")
    parser.add_argument("--filter-by-sascore", action="store_true")
    parser.add_argument("--sascore-threshold", default=0.7, type=float)
    parser.add_argument("--filter-by-docking-scores", action="store_true", 
                        help="Samples will be docked directly after generation and filtered versus a ground truth docking score. Only higher score will be kept.")
    parser.add_argument("--docking-scores", type=Path, default=None, 
                        help="If filter-by-docking-score is set to True, you have to provide ground-truth docking scores as a dictionary containing the respective ground truth ligand names and their scores")
    parser.add_argument("--omit-posebusters", default=False, action="store_true")
    parser.add_argument("--omit-posecheck", default=False, action="store_true")
    #load external models
    parser.add_argument("--ckpt-property-model", default=None, type=str)
    parser.add_argument("--ckpt-sa-model", default=None, type=str)
    parser.add_argument("--ckpts-ensemble", default=[], nargs="+", type=str)
    # classifier guidance
    parser.add_argument("--property-classifier-guidance", default=False, action="store_true")
    parser.add_argument("--property-classifier-self-guidance", default=False, action="store_true")
    parser.add_argument("--property-classifier-guidance-complex", default=False, action="store_true")
    parser.add_argument("--classifier-guidance-scale", default=1.e-4, type=float)
    # importance sampling
    parser.add_argument("--sa-importance-sampling", default=False, action="store_true")
    parser.add_argument("--sa-tau", default=0.1, type=float)
    parser.add_argument("--sa-every-importance-t", default=5, type=int)
    parser.add_argument("--sa-importance-sampling-start", default=None, type=int)
    parser.add_argument("--sa-importance-sampling-end", default=None, type=int)
    parser.add_argument("--minimize-property", default=False, action="store_true")
    parser.add_argument("--property-importance-sampling", default=False, action="store_true")
    parser.add_argument("--property-tau", default=0.1, type=float)
    parser.add_argument("--property-every-importance-t", default=5, type=int)
    parser.add_argument("--property-importance-sampling-start", default=None, type=int)
    parser.add_argument("--property-importance-sampling-end", default=None, type=int)
    parser.add_argument("--joint-importance-sampling", default=False, action="store_true")
    parser.add_argument("--property-normalization", default=False, action="store_true")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    evaluate(args)
