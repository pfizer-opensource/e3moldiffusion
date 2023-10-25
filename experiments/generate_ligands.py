import argparse
import warnings
from pathlib import Path
from time import time

import torch
from rdkit import Chem
from tqdm import tqdm

from experiments.utils import prepare_pocket, write_sdf_file
from Bio.PDB import PDBParser
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
    save_dir,
    test_dir,
    test_list,
    skip_existing,
    fix_n_nodes,
    ngraphs,
    batch_size,
    ddpm,
    eta_ddim,
):
    # load hyperparameter
    hparams = torch.load(model_path)["hyper_parameters"]
    hparams["select_train_subset"] = False
    hparams["diffusion_pretraining"] = False
    hparams = dotdict(hparams)

    hparams.load_ckpt_from_pretrained = None
    hparams.load_ckpt = None
    hparams.gpus = 1

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

    datamodule = DataModule(hparams)

    from experiments.data.data_info import GeneralInfos as DataInfos

    dataset_info = DataInfos(datamodule, hparams)

    train_smiles = list(datamodule.train_dataset.smiles)

    prop_norm, prop_dist = None, None
    if len(hparams.properties_list) > 0 and hparams.context_mapping:
        prop_norm = datamodule.compute_mean_mad(hparams.properties_list)
        prop_dist = DistributionProperty(datamodule, hparams.properties_list)
        prop_dist.set_normalizer(prop_norm)

    from experiments.diffusion_discrete_pocket import Trainer

    # if you want bond_model_guidance, flag this here in the Trainer
    device = "cuda"
    model = Trainer.load_from_checkpoint(
        model_path,
        dataset_info=dataset_info,
        smiles_list=train_smiles,
        prop_norm=prop_norm,
        prop_dist=prop_dist,
        load_ckpt_from_pretrained=None,
        load_ckpt=None,
        vdim=256,
        # energy_model_guidance=True if use_energy_guidance else False,
        # ckpt_energy_model=ckpt_energy_model,
        run_evaluation=True,
        strict=False,
    ).to(device)
    model = model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    save_dir.mkdir(exist_ok=skip_existing)
    raw_sdf_dir = Path(save_dir, "raw")
    raw_sdf_dir.mkdir(exist_ok=skip_existing)
    processed_sdf_dir = Path(save_dir, "processed")
    processed_sdf_dir.mkdir(exist_ok=skip_existing)
    times_dir = Path(save_dir, "pocket_times")
    times_dir.mkdir(exist_ok=skip_existing)

    test_files = list(test_dir.glob("[!.]*.sdf"))
    if test_list is not None:
        with open(test_list, "r") as f:
            test_list = set(f.read().split(","))
        test_files = [x for x in test_files if x.stem in test_list]

    pbar = tqdm(test_files)
    time_per_pocket = {}

    all_molecules = []
    for sdf_file in pbar:
        ligand_name = sdf_file.stem

        pdb_name, pocket_id, *suffix = ligand_name.split("_")
        pdb_file = Path(sdf_file.parent, f"{pdb_name}.pdb")
        txt_file = Path(sdf_file.parent, f"{ligand_name}.txt")
        sdf_out_file_raw = Path(raw_sdf_dir, f"{ligand_name}_gen.sdf")
        sdf_out_file_processed = Path(processed_sdf_dir, f"{ligand_name}_gen.sdf")
        time_file = Path(times_dir, f"{ligand_name}.txt")

        if (
            skip_existing
            and time_file.exists()
            and sdf_out_file_processed.exists()
            and sdf_out_file_raw.exists()
        ):
            with open(time_file, "r") as f:
                time_per_pocket[str(sdf_file)] = float(f.read().split()[1])

            continue

        t_pocket_start = time()

        with open(txt_file, "r") as f:
            resi_list = f.read().split()

        if fix_n_nodes:
            # some ligands (e.g. 6JWS_bio1_PT1:A:801) could not be read with sanitize=True
            suppl = Chem.SDMolSupplier(str(sdf_file), sanitize=False)
            num_nodes_lig = suppl[0].GetNumAtoms()

        pdb_struct = PDBParser(QUIET=True).get_structure("", pdb_file)[0]
        if resi_list is not None:
            # define pocket with list of residues
            residues = [
                pdb_struct[x.split(":")[0]][(" ", int(x.split(":")[1]), " ")]
                for x in resi_list
            ]
        pocket_data = prepare_pocket(
            residues, dataset_info.atom_encoder, no_H=True, repeats=1, device=device
        )

        molecules = model.generate_ligands(
            pocket_data,
            num_graphs=1,
            inner_verbose=False,
            save_traj=False,
            ddpm=ddpm,
            eta_ddim=eta_ddim,
            num_nodes_lig=num_nodes_lig,
            mol_device="cpu",
        )
        all_molecules.extend(molecules)
        write_sdf_file(sdf_out_file_raw, molecules)

        # Time the sampling process
        time_per_pocket[str(sdf_file)] = time() - t_pocket_start
        with open(time_file, "w") as f:
            f.write(f"{str(sdf_file)} {time_per_pocket[str(sdf_file)]}")

        pbar.set_description(
            f"Last processed: {ligand_name}. "
            f"{(time() - t_pocket_start) / len(all_molecules):.2f} "
            f"sec/mol."
        )

    with open(Path(save_dir, "pocket_times.txt"), "w") as f:
        for k, v in time_per_pocket.items():
            f.write(f"{k} {v}\n")

    times_arr = torch.tensor([x for x in time_per_pocket.values()])
    print(
        f"Time per pocket: {times_arr.mean():.3f} \pm "
        f"{times_arr.std(unbiased=False):.2f}"
    )


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description='Data generation')
    parser.add_argument('--model-path', default="/hpfs/userws/cremej01/workspace/logs/aqm_qm7x/x0_t_weighting_dip_mpol/best_mol_stab.ckpt", type=str,
                        help='Path to trained model')
    parser.add_argument("--use-energy-guidance", default=False, action="store_true")
    parser.add_argument("--ckpt-energy-model", default=None, type=str)
    parser.add_argument('--guidance-scale', default=1.0e-4, type=float,
                        help='How to scale the guidance shift')
    parser.add_argument('--save-xyz', default=False, action="store_true",
                        help='Whether or not to store generated molecules in xyz files')
    parser.add_argument('--calculate-energy', default=False, action="store_true",
                        help='Whether or not to calculate xTB energies and forces')
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
    parser.add_argument("--test-dir", type=Path)
    parser.add_argument("--test-list", type=Path, default=None)
    parser.add_argument("--save-dir", type=Path)
    parser.add_argument("--fix-n-nodes", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    # Evaluate negative log-likelihood for the test partitions
    evaluate(
        model_path=args.model_path,
        test_dir=args.test_dir,
        test_list=args.test_list,
        save_dir=args.save_dir,
        skip_existing=args.skip_existing,
        fix_n_nodes=args.fix_n_nodes,
        ngraphs=args.ngraphs,
        batch_size=args.batch_size,
        ddpm=not args.ddim,
        eta_ddim=args.eta_ddim,
    )
