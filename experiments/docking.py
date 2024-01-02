import argparse
import os
import re
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from posebusters import PoseBusters
from posecheck.posecheck import PoseCheck
from rdkit import Chem
from tqdm import tqdm

from experiments.data.ligand.process_pdb import get_pdb_components, write_pdb
from experiments.data.utils import save_pickle
from experiments.utils import retrieve_interactions_per_mol, write_sdf_file


def calculate_smina_score(pdb_file, sdf_file):
    # add '-o <name>_smina.sdf' if you want to see the output
    out = os.popen(f"smina.static -l {sdf_file} -r {pdb_file} " f"--score_only").read()
    matches = re.findall(r"Affinity:[ ]+([+-]?[0-9]*[.]?[0-9]+)[ ]+\(kcal/mol\)", out)
    return [float(x) for x in matches]


def smina_score(rdmols, receptor_file):
    """
    Calculate smina score
    :param rdmols: List of RDKit molecules
    :param receptor_file: Receptor pdb/pdbqt file or list of receptor files
    :return: Smina score for each input molecule (list)
    """

    if isinstance(receptor_file, list):
        scores = []
        for mol, rec_file in zip(rdmols, receptor_file):
            with tempfile.NamedTemporaryFile(suffix=".sdf") as tmp:
                tmp_file = tmp.name
                write_sdf_file(tmp_file, [mol])
                scores.extend(calculate_smina_score(rec_file, tmp_file))

    # Use same receptor file for all molecules
    else:
        with tempfile.NamedTemporaryFile(suffix=".sdf") as tmp:
            tmp_file = tmp.name
            write_sdf_file(tmp_file, rdmols)
            scores = calculate_smina_score(receptor_file, tmp_file)

    return scores


def sdf_to_pdbqt(sdf_file, pdbqt_outfile, mol_id):
    os.popen(
        f"obabel {sdf_file} -O {pdbqt_outfile} " f"-f {mol_id + 1} -l {mol_id + 1}"
    ).read()
    return pdbqt_outfile


def calculate_qvina2_score(
    receptor_file,
    sdf_file,
    out_dir,
    pdb_dir,
    buster_dict,
    violin_dict,
    posecheck_dict,
    size=20,
    exhaustiveness=16,
    return_rdmol=False,
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
    pdb_dir = Path(pdb_dir)

    if receptor_file.suffix == ".pdb":
        # prepare receptor, requires Python 2.7
        receptor_pdbqt_file = Path(out_dir, receptor_file.stem + ".pdbqt")
        os.popen(f"prepare_receptor4.py -r {receptor_file} -O {receptor_pdbqt_file}")
    else:
        receptor_pdbqt_file = receptor_file

    scores = []
    rdmols = []  # for if return rdmols
    clashes = []
    strain_energies = []
    rmsds = []

    suppl = Chem.SDMolSupplier(str(sdf_file), sanitize=False)
    ligand_name = sdf_file.stem
    ligand_pdbqt_file = Path(out_dir, ligand_name + ".pdbqt")
    out_sdf_file = Path(out_dir, ligand_name + "_out.sdf")

    # Initialize the PoseCheck object
    pc = PoseCheck()

    pdb_name = str(ligand_pdbqt_file).split("/")[-1].split("-")[0]
    if pdb_dir is not None:
        pdbs = [str(i).split("/")[-1].split(".")[0] for i in pdb_dir.glob("[!.]*.pdb")]
    else:
        pdbs = None
    if pdbs is not None and pdb_name in pdbs:
        pdb_file = os.path.join(str(pdb_dir), pdb_name + ".pdb")
    else:
        temp_dir = tempfile.mkdtemp()
        protein, _ = get_pdb_components(pdb_name)
        pdb_file = write_pdb(temp_dir, protein, pdb_name)

    pc.load_protein_from_pdb(pdb_file)

    for i, mol in enumerate(suppl):  # sdf file may contain several ligands
        sdf_to_pdbqt(sdf_file, ligand_pdbqt_file, i)

        # center box at ligand's center of mass
        cx, cy, cz = mol.GetConformer().GetPositions().mean(0)

        # run QuickVina 2
        try:
            os.stat("/sharedhome/cremej01/workspace/e3moldiffusion/qvina2.1")
            PATH = "/sharedhome/cremej01/workspace/e3moldiffusion/qvina2.1"
        except PermissionError:
            PATH = "/sharedhome/let55/projects/e3moldiffusion/qvina2.1"

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
            scores.append(np.nan)
            continue

        out_split = out.splitlines()
        best_idx = out_split.index("-----+------------+----------+----------") + 1
        best_line = out_split[best_idx].split()
        assert best_line[0] == "1"
        scores.append(float(best_line[1]))

        out_pdbqt_file = Path(out_dir, ligand_name + "_out.pdbqt")
        if out_pdbqt_file.exists():
            os.popen(f"obabel {out_pdbqt_file} -O {out_sdf_file}").read()
            # clean up
            out_pdbqt_file.unlink()

        rdmol = Chem.SDMolSupplier(str(out_sdf_file))[0]
        if rdmol is None:
            continue
        rdmols.append(rdmol)

        pc.load_ligands_from_sdf(str(out_sdf_file), add_hs=True)
        # pc.load_ligands_from_mol(rdmol)
        clashes.append(pc.calculate_clashes()[0])
        strain_energies.append(pc.calculate_strain_energy()[0])
        rmsds.append(pc.calculate_rmsd(suppl[i], rdmol))

    posecheck_dict["Clashes"].append(np.mean(clashes))
    posecheck_dict["Strain Energies"].append(np.mean(strain_energies))
    posecheck_dict["RMSD"].append(np.mean(rmsds))

    violin_dict["clashes"].extend(clashes)
    violin_dict["strain_energy"].extend(strain_energies)
    violin_dict["rmsd"].extend(rmsds)

    write_sdf_file(out_sdf_file, rdmols)

    # PoseBusters
    print("Starting evaluation with PoseBusters...")
    buster = {}
    buster_mol = PoseBusters(config="mol")
    buster_mol_df = buster_mol.bust([out_sdf_file], None, None)
    for metric in buster_mol_df.columns:
        violin_dict[metric].extend(list(buster_mol_df[metric]))
        buster[metric] = buster_mol_df[metric].sum() / len(buster_mol_df[metric])
    buster_dock = PoseBusters(config="dock")
    buster_dock_df = buster_dock.bust([out_sdf_file], None, pdb_file)
    for metric in buster_dock_df:
        if metric not in buster:
            violin_dict[metric].extend(list(buster_dock_df[metric]))
            buster[metric] = buster_dock_df[metric].sum() / len(buster_dock_df[metric])
    for k, v in buster.items():
        buster_dict[k].append(v)
    print("Done!")

    # # PoseCheck
    # print("Starting evaluation with PoseCheck...")
    # pc = PoseCheck()
    # pc.load_protein_from_pdb(pdb_file)
    # pc.load_ligands_from_mols(rdmols, add_hs=True)
    # interactions = pc.calculate_interactions()
    # interactions_per_mol, interactions_mean = retrieve_interactions_per_mol(
    #     interactions
    # )
    # for k, v in interactions_per_mol.items():
    #     violin_dict[k].extend(v)
    # for k, v in interactions_mean.items():
    #     posecheck_dict[k].append(v["mean"])
    # posecheck_dict["Clashes"].append(np.mean(pc.calculate_clashes()))
    # posecheck_dict["Strain Energies"].append(np.mean(pc.calculate_strain_energy()))
    # print("Done!")

    try:
        shutil.rmtree(temp_dir)
    except UnboundLocalError:
        pass

    if return_rdmol:
        return scores, rdmols
    else:
        return scores, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser("QuickVina evaluation")
    parser.add_argument("--pdbqt-dir", type=Path, help="Receptor files in pdbqt format")
    parser.add_argument(
        "--sdf-dir", type=Path, default=None, help="Ligand files in sdf format"
    )
    parser.add_argument("--sdf-files", type=Path, nargs="+", default=None)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument(
        "--pdb-dir",
        type=Path,
        default=None,
        help="Directory where all full protein pdb files are stored. If not available, there will be calculated on the fly.",
    )
    parser.add_argument("--write-csv", action="store_true")
    parser.add_argument("--write-dict", action="store_true")
    parser.add_argument("--dataset", type=str, default="moad")
    args = parser.parse_args()

    print("Starting docking...")

    assert (args.sdf_dir is not None) ^ (args.sdf_files is not None)

    args.out_dir.mkdir(exist_ok=True)

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
        elif args.dataset == "crossdocked":
            ligand_name = sdf_file.stem
            receptor_name = ligand_name.split("_")[0]
            receptor_file = Path(args.pdbqt_dir, receptor_name + ".pdbqt")
        elif args.dataset == "cdk2":
            ligand_name = sdf_file.stem
            receptor_name = ligand_name.split("_")[0]
            receptor_file = Path(args.pdbqt_dir, receptor_name + ".pdbqt")

        # try:
        scores, rdmols = calculate_qvina2_score(
            receptor_file,
            sdf_file,
            args.out_dir,
            args.pdb_dir,
            buster_dict,
            violin_dict,
            posecheck_dict,
            return_rdmol=True,
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

    if args.write_csv:
        df = pd.DataFrame.from_dict(results)
        df.to_csv(Path(args.out_dir, "qvina2_scores.csv"))

    if args.write_dict:
        torch.save(results_dict, Path(args.out_dir, "qvina2_scores.pt"))
        save_pickle(buster_dict, os.path.join(args.out_dir, "posebusters.pickle"))
        save_pickle(violin_dict, os.path.join(args.out_dir, "violin_dict.pickle"))
        save_pickle(posecheck_dict, os.path.join(args.out_dir, "posecheck.pickle"))

    scores_mean = [np.mean(r) for r in results["scores"] if len(r) >= 1]

    missing = len(results["scores"]) - len(scores_mean)
    print(f"Number of dockings evaluated with NaN: {missing}")

    # Scores
    mean_score = np.mean(scores_mean)
    std_score = np.std(scores_mean)
    print(f"Mean docking score: {mean_score}")
    print(f"Docking score standard deviation: {std_score}")

    # PoseBusters
    buster_dict = {
        k: {"mean": np.mean(v), "std": np.std(v)} for k, v in buster_dict.items()
    }
    print(f"PoseBusters evaluation: {buster_dict}")

    # PoseCheck
    posecheck_dict = {
        k: {"mean": np.mean(v), "std": np.std(v)} for k, v in posecheck_dict.items()
    }
    print(f"PoseCheck evaluation: {posecheck_dict}")

    # scores = np.mean(
    #     [r.sort(reverse=True)[:10] for r in results["scores"] if len(r) >= 1]
    # )
    # mean_top10_score = np.mean(scores)
    # print(f"Top-10 mean score: {mean_top10_score}")
