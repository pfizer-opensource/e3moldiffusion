import argparse
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from posebusters import PoseBusters
from posecheck.posecheck import PoseCheck
from rdkit import Chem
from tqdm import tqdm

from experiments.data.utils import save_pickle
from experiments.utils import retrieve_interactions_per_mol, split_list, write_sdf_file


def calculate_qvina2_score(
    pdb_file,
    receptor_file,
    sdf_file,
    out_dir,
    buster_dict,
    violin_dict,
    posecheck_dict,
    size=20,
    exhaustiveness=16,
    return_rdmol=False,
    run_eval=True,
):
    """
    Calculate the QuickVina2 score

    Parameters:
    - receptor_file (str): The receptor in pdbqt-format
    - sdf_file (str): The ligand(s) in sdf-format
    - out_dir (str): The directory the docked molecules shall be saved to

    Returns:
    - tuple: (docking scores, PoseCheck dictionary, RDKit molecules [docked])

    """

    receptor_file = Path(receptor_file)
    sdf_file = Path(sdf_file)
    pdb_file = Path(pdb_file)

    if receptor_file.suffix == ".pdb":
        # prepare receptor, requires Python 2.7
        receptor_pdbqt_file = Path(
            os.path.join(out_dir, "docked"), receptor_file.stem + ".pdbqt"
        )
        os.popen(f"prepare_receptor4.py -r {receptor_file} -O {receptor_pdbqt_file}")
    else:
        receptor_pdbqt_file = receptor_file

    scores = []
    rdmols = []  # for if return rdmols

    suppl = Chem.SDMolSupplier(str(sdf_file), sanitize=False)
    ligand_name = sdf_file.stem
    ligand_pdbqt_file = Path(os.path.join(out_dir, "docked"), ligand_name + ".pdbqt")
    out_sdf_file = Path(os.path.join(out_dir, "docked"), ligand_name + "_out.sdf")

    for i, mol in enumerate(suppl):  # sdf file may contain several ligands
        sdf_to_pdbqt(sdf_file, ligand_pdbqt_file, i)

        # center box at ligand's center of mass
        cx, cy, cz = mol.GetConformer().GetPositions().mean(0)

        # run QuickVina 2
        try:
            os.stat("/sharedhome/cremej01/workspace/qvina2.1")
            PATH = "/sharedhome/cremej01/workspace/qvina2.1"
        except FileNotFoundError:
            os.stat("/hpfs/userws/cremej01/projects/qvina2.1")
            PATH = "/hpfs/userws/cremej01/projects/qvina2.1"
        except PermissionError:
            PATH = "/sharedhome/let55/projects/e3moldiffusion/qvina2.1"
            # PATH = "/hpfs/userws/let55/projects/e3moldiffusion/qvina2.1"

        out = os.popen(
            f"/{PATH} --receptor {receptor_pdbqt_file} "
            f"--ligand {ligand_pdbqt_file} "
            f"--center_x {cx:.4f} --center_y {cy:.4f} --center_z {cz:.4f} "
            f"--size_x {size} --size_y {size} --size_z {size} "
            f"--exhaustiveness {exhaustiveness}",
        ).read()
        # clean up
        ligand_pdbqt_file.unlink()

        if "-----+------------+----------+----------" not in out:
            continue

        out_split = out.splitlines()
        best_idx = out_split.index("-----+------------+----------+----------") + 1
        best_line = out_split[best_idx].split()
        assert best_line[0] == "1"
        scores.append(float(best_line[1]))

        out_pdbqt_file = Path(
            os.path.join(out_dir, "docked"), ligand_name + "_out.pdbqt"
        )
        if out_pdbqt_file.exists():
            os.popen(f"obabel {out_pdbqt_file} -O {out_sdf_file}").read()
            # clean up
            out_pdbqt_file.unlink()

        rdmol = Chem.SDMolSupplier(str(out_sdf_file))[0]

        if rdmol is None:
            continue
        rdmols.append(rdmol)

    if len(rdmols) > 0 and run_eval:
        write_sdf_file(out_sdf_file, rdmols)

        # PoseBusters
        print("Starting evaluation with PoseBusters...")
        buster = {}
        buster_mol = PoseBusters(config="mol")
        buster_mol_df = buster_mol.bust([str(out_sdf_file)], None, None)
        for metric in buster_mol_df.columns:
            violin_dict[metric].extend(list(buster_mol_df[metric]))
            buster[metric] = buster_mol_df[metric].sum() / len(buster_mol_df[metric])
        buster_dock = PoseBusters(config="dock")
        buster_dock_df = buster_dock.bust([str(out_sdf_file)], None, str(pdb_file))
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
        pc.load_ligands_from_mols(rdmols, add_hs=True)
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

    if return_rdmol:
        return scores, rdmols
    else:
        return scores, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser("QuickVina evaluation")
    parser.add_argument("--mp-index", default=0, type=int)
    parser.add_argument("--num-cpus", default=20, type=int)
    parser.add_argument("--pdbqt-dir", type=Path, help="Receptor files in pdbqt format")
    parser.add_argument(
        "--sdf-dir", type=Path, default=None, help="Ligand files in sdf format"
    )
    parser.add_argument("--sdf-files", type=Path, nargs="+", default=None)
    parser.add_argument("--save-dir", type=Path)
    parser.add_argument(
        "--pdb-dir",
        type=Path,
        default=None,
        help="Directory where all full protein pdb files are stored. If not available, there will be calculated on the fly.",
    )
    parser.add_argument("--write-csv", action="store_true")
    parser.add_argument("--write-dict", action="store_true")
    parser.add_argument("--avoid-eval", default=False, action="store_true")
    parser.add_argument("--dataset", type=str, default="moad")
    args = parser.parse_args()

    print("Starting docking...")

    assert (args.sdf_dir is not None) ^ (args.sdf_files is not None)

    results = {"receptor": [], "ligand": [], "scores": []}
    results_dict = {}

    buster_dict = defaultdict(list)
    violin_dict = defaultdict(list)
    posecheck_dict = defaultdict(list)

    sdf_files = (
        list(args.sdf_dir.glob("[!.]*.sdf"))
        if args.sdf_dir is not None
        else args.sdf_files
    )

    # assert len(sdf_files) == len(glob(os.path.join(args.pdb_dir, ".pdb")))

    assert np.sum([len(i) for i in split_list(sdf_files, args.num_cpus)]) == len(
        sdf_files
    )
    sdf_files = split_list(sdf_files, args.num_cpus)[args.mp_index - 1]

    pbar = tqdm(sdf_files)
    for sdf_file in pbar:
        pbar.set_description(f"Processing {sdf_file.name}")

        if args.dataset == "moad":
            """
            Ligand file names should be of the following form:
            <receptor-name>_<pocket-id>_<some-suffix>.sdf
            where <receptor-name> and <pocket-id> cannot contain any
            underscores, e.g.: 1abc-bio1_pocket0_gen.sdf
            """
            ligand_name = sdf_file.stem
            receptor_name, pocket_id, *suffix = ligand_name.split("_")
            suffix = "_".join(suffix)
            receptor_file = Path(args.pdbqt_dir, receptor_name + ".pdbqt")
            pdb_file = Path(args.pdb_dir, receptor_name + ".pdb")
        elif args.dataset == "crossdocked":
            ligand_name = sdf_file.stem
            receptor_name = ligand_name.split("_")[0]
            receptor_file = Path(args.pdbqt_dir, receptor_name + ".pdbqt")
            pdb_file = Path(args.pdb_dir, receptor_name + ".pdb")
        elif args.dataset == "cdk2":
            ligand_name = sdf_file.stem
            receptor_name = ligand_name.split("_")[0]
            receptor_file = Path(args.pdbqt_dir, receptor_name + ".pdbqt")
            pdb_file = Path(args.pdb_dir, receptor_name + ".pdb")
        elif args.dataset == "kinodata":
            ligand_name = sdf_file.stem
            receptor_name = ligand_name.split("_")[0]
            receptor_file = Path(args.pdbqt_dir, receptor_name + ".pdbqt")
            pdb_file = Path(args.pdb_dir, receptor_name + ".pdb")
        elif args.dataset == "molecular_glue":
            ligand_name = ("_").join(sdf_file.stem.split("_")[1:])
            receptor_name = ligand_name.split("_")[0]
            receptor_file = Path(args.pdbqt_dir, receptor_name + ".pdbqt")
            pdb_file = Path(args.pdb_dir, receptor_name + ".pdb")
        else:
            raise Exception("Dataset not available!")

        # try:
        scores, rdmols = calculate_qvina2_score(
            pdb_file,
            receptor_file,
            sdf_file,
            args.save_dir,
            buster_dict,
            violin_dict,
            posecheck_dict,
            return_rdmol=True,
            run_eval=not args.avoid_eval,
        )
        # except AttributeError as e:
        #     print(e)
        #     continue
        results["receptor"].append(str(receptor_file))
        results["ligand"].append(str(sdf_file))
        results["scores"].append(scores)

        if args.write_dict:
            results_dict[ligand_name] = {
                "receptor": str(receptor_file),
                "ligand": str(sdf_file),
                "scores": scores,
                "rmdols": rdmols,
            }

    if args.write_dict:
        save_pickle(
            results,
            os.path.join(args.save_dir, f"{args.mp_index}_qvina2_scores.pickle"),
        )
        save_pickle(
            buster_dict,
            os.path.join(args.save_dir, f"{args.mp_index}_posebusters_docked.pickle"),
        )
        save_pickle(
            violin_dict,
            os.path.join(args.save_dir, f"{args.mp_index}_violin_dict_docked.pickle"),
        )
        save_pickle(
            posecheck_dict,
            os.path.join(args.save_dir, f"{args.mp_index}_posecheck_docked.pickle"),
        )
