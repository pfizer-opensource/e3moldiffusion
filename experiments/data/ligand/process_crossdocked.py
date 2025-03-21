import argparse
import random
import shutil
import subprocess
from pathlib import Path
from time import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa
from Bio.PDB.Polypeptide import protein_letters_3to1 as three_to_one
from rdkit import Chem
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

from experiments.data.ligand import constants
from experiments.data.ligand.constants import (
    atom_decoder_int,
    covalent_radii,
    dataset_params,
)
from experiments.data.ligand.molecule_builder import build_molecule
from experiments.data.utils import load_pickle
from posecheck.utils.constants import REDUCE_PATH

dataset_info = dataset_params["crossdock_full"]
amino_acid_dict = dataset_info["aa_encoder"]
aa_atom_encoder = dataset_info["aa_atom_encoder"]
atom_dict = dataset_info["atom_encoder"]
atom_decoder = dataset_info["atom_decoder"]


def process_ligand_and_pocket(
    pdbfile,
    sdffile,
    dist_cutoff,
    ca_only,
    no_H,
    reduce_path=REDUCE_PATH,
    complex_creation_mode="diffsbdd",
    com_weights: bool = False,
):
    if not no_H:
        tmp_path = str(pdbfile).split(".pdb")[0] + "_tmp.pdb"
        # Call reduce to make tmp PDB with waters
        reduce_command = f"{reduce_path} -NOFLIP  {pdbfile} -Quiet > {tmp_path}"
        subprocess.run(reduce_command, shell=True)
        pdb_struct = PDBParser(QUIET=True).get_structure("", tmp_path)
    else:
        pdb_struct = PDBParser(QUIET=True).get_structure("", pdbfile)

    try:
        ligand = Chem.SDMolSupplier(str(sdffile), removeHs=no_H)[0]
        if not no_H:
            ligand = Chem.AddHs(ligand, addCoords=True)
    except:
        raise Exception(f"cannot read sdf mol ({sdffile})")

    # remove H atoms if not in atom_dict, other atom types that aren't allowed
    # should stay so that the entire ligand can be removed from the dataset
    lig_atoms = [a.GetSymbol() for a in ligand.GetAtoms()]
    lig_coords = np.array(
        [
            list(ligand.GetConformer(0).GetAtomPosition(idx))
            for idx in range(ligand.GetNumAtoms())
        ]
    )

    # Find interacting pocket residues based on distance cutoff

    # diffsbdd: Iterate over _all_ atoms in a residue and compute pairwise distance between them and all ligand atoms. If one of the distances is smaller than the cutoff, then the entire residue is considered to be in the pocket. (Hence doing min selection)
    # targediff: Choose the CoM of the atoms in a residue as representative and compute pairwise distance between the representative and all ligand atoms. If the distance is smaller than the cutoff, then the entire residue is considered to be in the pocket.
    pocket_residues = []
    for residue in pdb_struct[0].get_residues():
        res_coords = np.array([a.get_coord() for a in residue.get_atoms()])
        if complex_creation_mode == "targetdiff":
            # https://biopython.org/docs/dev/api/Bio.PDB.Entity.html#Bio.PDB.Entity.Entity.center_of_mass
            # https://github.com/guanjq/targetdiff/blob/main/utils/data.py#L130-L138
            res_coords = residue.center_of_mass(geometric=not com_weights)
            res_coords = res_coords[np.newaxis, :]

        if (
            is_aa(residue.get_resname(), standard=True)
            and (
                ((res_coords[:, None, :] - lig_coords[None, :, :]) ** 2).sum(-1) ** 0.5
            ).min()
            < dist_cutoff
        ):
            pocket_residues.append(residue)

    pocket_chainids = [f"{res.id[1]}.{res.parent.id}" for res in pocket_residues]
    pocket_resnames = [res.get_resname() for res in pocket_residues]
    pocket_resid = [res.id[1] for res in pocket_residues]
    ligand_data = {
        "lig_coords": lig_coords,
        "lig_atoms": lig_atoms,
        "lig_mol": ligand,
    }
    if ca_only:
        try:
            pocket_one_hot = []
            full_coords = []
            for res in pocket_residues:
                for atom in res.get_atoms():
                    if atom.name == "CA":
                        pocket_one_hot.append(
                            np.eye(
                                1,
                                len(amino_acid_dict),
                                amino_acid_dict[three_to_one.get(res.get_resname())],
                            ).squeeze()
                        )
                        full_coords.append(atom.coord)
            pocket_one_hot = np.stack(pocket_one_hot)
            full_coords = np.stack(full_coords)
        except KeyError as e:
            raise KeyError(f"{e} not in amino acid dict ({pdbfile}, {sdffile})")
        pocket_data = {
            "pocket_coords": full_coords,
            "pocket_one_hot": pocket_one_hot,
            "pocket_chainids": pocket_chainids,
        }
    else:
        # c-alphas and residue idendity
        pocket_one_hot = []
        ca_mask = []

        # full
        full_atoms = []
        full_atom_names = []
        full_coords = []
        m = False
        for res in pocket_residues:
            for atom in res.get_atoms():
                if atom.name == "CA":
                    pocket_one_hot.append(
                        np.eye(
                            1,
                            len(amino_acid_dict),
                            amino_acid_dict[three_to_one.get(res.get_resname())],
                        ).squeeze()
                    )
                    m = True
                else:
                    m = False
                ca_mask.append(m)
                full_atoms.append(atom.element)
                full_atom_names.append(atom.name)
                full_coords.append(atom.coord)

        pocket_one_hot = np.stack(pocket_one_hot, axis=0)
        full_atoms = np.stack(full_atoms, axis=0)
        full_atom_names = np.stack(full_atom_names, axis=0)
        full_coords = np.stack(full_coords, axis=0)
        ca_mask = np.array(ca_mask, dtype=bool)
        if no_H:
            indices_H = np.where(full_atoms == "H")
            if indices_H[0].size > 0:
                mask = np.ones(full_atoms.size, dtype=bool)
                mask[indices_H] = False
                full_atoms = full_atoms[mask]
                full_atom_names = full_atom_names[mask]
                full_coords = full_coords[mask]
                ca_mask = ca_mask[mask]

        assert sum(ca_mask) == pocket_one_hot.shape[0]
        assert len(full_atoms) == len(full_coords)
        pocket_data = {
            "pocket_coords": full_coords,
            "pocket_resnames": pocket_resnames,
            "pocket_chainids": pocket_chainids,
            "pocket_resids": pocket_resid,
            "pocket_atoms": full_atoms,
            "pocket_atom_names": full_atom_names,
            "pocket_one_hot": pocket_one_hot,
            "pocket_ca_mask": ca_mask,
        }
    return ligand_data, pocket_data


def compute_smiles(positions, atom_types, mask):
    print("Computing SMILES ...")

    sections = np.where(np.diff(mask))[0] + 1
    positions = [torch.from_numpy(x) for x in np.split(positions, sections)]
    atom_types = [
        torch.tensor([atom_dict[a] for a in atoms])
        for atoms in np.split(atom_types, sections)
    ]

    mols_smiles = []
    fail = 0
    pbar = tqdm(enumerate(zip(positions, atom_types)), total=len(np.unique(mask)))
    for i, (pos, atom_type) in pbar:
        atom_type = [atom_decoder_int[int(a)] for a in atom_type]
        mol = build_molecule(pos, atom_type, dataset_info)
        try:
            mol = Chem.MolToSmiles(mol)
        except:
            fail += 1
            continue
        if mol is not None:
            mols_smiles.append(mol)
        pbar.set_description(
            f"{len(mols_smiles)}/{i + 1} successful, {len(mols_smiles)}/{fail} fail"
        )

    return mols_smiles


def get_n_nodes(lig_mask, pocket_mask, smooth_sigma=None):
    # Joint distribution of ligand's and pocket's number of nodes
    idx_lig, n_nodes_lig = np.unique(lig_mask, return_counts=True)
    idx_pocket, n_nodes_pocket = np.unique(pocket_mask, return_counts=True)
    assert np.all(idx_lig == idx_pocket)

    joint_histogram = np.zeros((np.max(n_nodes_lig) + 1, np.max(n_nodes_pocket) + 1))

    for nlig, npocket in zip(n_nodes_lig, n_nodes_pocket):
        joint_histogram[nlig, npocket] += 1

    print(
        f"Original histogram: {np.count_nonzero(joint_histogram)}/"
        f"{joint_histogram.shape[0] * joint_histogram.shape[1]} bins filled"
    )

    # Smooth the histogram
    if smooth_sigma is not None:
        filtered_histogram = gaussian_filter(
            joint_histogram,
            sigma=smooth_sigma,
            order=0,
            mode="constant",
            cval=0.0,
            truncate=4.0,
        )

        print(
            f"Smoothed histogram: {np.count_nonzero(filtered_histogram)}/"
            f"{filtered_histogram.shape[0] * filtered_histogram.shape[1]} bins filled"
        )

        joint_histogram = filtered_histogram

    return joint_histogram


def get_bond_length_arrays(atom_mapping):
    bond_arrays = []
    for i in range(3):
        bond_dict = getattr(constants, f"bonds{i + 1}")
        bond_array = np.zeros((len(atom_mapping), len(atom_mapping)))
        for a1 in atom_mapping.keys():
            for a2 in atom_mapping.keys():
                if a1 in bond_dict and a2 in bond_dict[a1]:
                    bond_len = bond_dict[a1][a2]
                else:
                    bond_len = 0
                bond_array[atom_mapping[a1], atom_mapping[a2]] = bond_len

        assert np.all(bond_array == bond_array.T)
        bond_arrays.append(bond_array)

    return bond_arrays


def get_lennard_jones_rm(atom_mapping):
    # Bond radii for the Lennard-Jones potential
    LJ_rm = np.zeros((len(atom_mapping), len(atom_mapping)))

    for a1 in atom_mapping.keys():
        for a2 in atom_mapping.keys():
            all_bond_lengths = []
            for btype in ["bonds1", "bonds2", "bonds3"]:
                bond_dict = getattr(constants, btype)
                if a1 in bond_dict and a2 in bond_dict[a1]:
                    all_bond_lengths.append(bond_dict[a1][a2])

            if len(all_bond_lengths) > 0:
                # take the shortest possible bond length because slightly larger
                # values aren't penalized as much
                bond_len = min(all_bond_lengths)
            else:
                if a1 == "others" or a2 == "others":
                    bond_len = 0
                else:
                    # Replace missing values with sum of average covalent radii
                    bond_len = covalent_radii[a1] + covalent_radii[a2]

            LJ_rm[atom_mapping[a1], atom_mapping[a2]] = bond_len

    assert np.all(LJ_rm == LJ_rm.T)
    return LJ_rm


def get_type_histograms(lig_atoms, pocket_atom, atom_encoder, aa_encoder):
    atom_counts = {k: 0 for k in atom_encoder.keys()}
    for a in lig_atoms:
        atom_counts[a] += 1

    aa_counts = {k: 0 for k in aa_encoder.keys()}
    for r in pocket_atom:
        aa_counts[r] += 1

    return atom_counts, aa_counts


def saveall(
    filename,
    pdb_and_mol_ids,
    lig_coords,
    lig_atom,
    lig_mask,
    lig_mol,
    pocket_coords,
    pocket_atom,
    pocket_atom_names,
    pocket_mask,
    pocket_resids,
    pocket_chainids,
    pocket_resnames,
    pocket_one_hot,
    pocket_ca_mask,
    docking_scores,
):
    np.savez(
        filename,
        names=pdb_and_mol_ids,
        lig_coords=lig_coords,
        lig_atom=lig_atom,
        lig_mask=lig_mask,
        lig_mol=lig_mol,
        pocket_coords=pocket_coords,
        pocket_atom=pocket_atom,
        pocket_atom_names=pocket_atom_names,
        pocket_mask=pocket_mask,
        pocket_resids=pocket_resids,
        pocket_chainids=pocket_chainids,
        pocket_resnames=pocket_resnames,
        pocket_one_hot=pocket_one_hot,
        pocket_ca_mask=pocket_ca_mask,
        docking_scores=docking_scores,
    )
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--basedir", type=Path)
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--no-H", action="store_true")
    parser.add_argument("--ca-only", action="store_true")
    parser.add_argument("--dist-cutoff", type=float, default=8.0)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--with-docking-scores", action="store_true")
    parser.add_argument("--path-docking-scores", type=str, default=None)
    parser.add_argument(
        "--complex-creation-mode",
        type=str,
        default="diffsbdd",
        help="Mode for creating complexes, either 'diffsbdd' or 'targetdiff'. Defaults to 'diffsbdd'.",
    )
    parser.add_argument(
        "--com-weights",
        action="store_true",
        help="If the center of mass should be obtained through weighted average of atoms in a residue including atomic weights. Defaults to False",
    )

    args = parser.parse_args()

    datadir = args.basedir / "crossdocked_pocket10/"

    if args.ca_only:
        dataset_info = dataset_params["crossdock"]

    # Make output directory
    if args.outdir is None:
        suffix = "_crossdock_noH" if args.no_H else "_crossdock_H"
        suffix += "_ca_only_temp" if args.ca_only else "_full_temp"
        suffix += f"_cutoff{args.dist_cutoff}"
        processed_dir = Path(args.basedir, f"processed{suffix}")
    else:
        processed_dir = args.outdir

    processed_dir.mkdir(exist_ok=True, parents=True)

    # Read data split
    split_path = Path(args.basedir, "split_by_name.pt")
    data_split = torch.load(split_path)

    # There is no validation set, copy 300 training examples (the validation set
    # is not very important in this application)
    # Note: before we had a data leak but it should not matter too much as most
    # metrics monitored during training are independent of the pockets
    data_split["val"] = random.sample(data_split["train"], 300)

    n_train_before = len(data_split["train"])
    n_val_before = len(data_split["val"])
    n_test_before = len(data_split["test"])

    failed_save = []

    if args.with_docking_scores:
        recs = load_pickle(args.path_docking_scores)

    n_samples_after = {}
    for split in data_split.keys():
        lig_coords = []
        lig_atom = []
        lig_mask = []
        lig_mol = []
        docking_scores = []
        pocket_coords = []
        pocket_atom = []
        pocket_atom_names = []
        pocket_mask = []
        pocket_resids = []
        pocket_chainids = []
        pocket_resnames = []
        # new
        pocket_one_hot_resids = []
        pocket_ca_mask = []
        pdb_and_mol_ids = []
        count_protein = []
        count_ligand = []
        count_total = []
        count = 0

        pdb_sdf_dir = processed_dir / split
        pdb_sdf_dir.mkdir(exist_ok=True)

        tic = time()
        num_failed = 0
        pbar = tqdm(data_split[split])
        pbar.set_description(f"#failed: {num_failed}")
        for pocket_fn, ligand_fn in pbar:
            sdffile = datadir / f"{ligand_fn}"
            pdbfile = datadir / f"{pocket_fn}"

            try:
                struct_copy = PDBParser(QUIET=True).get_structure("", pdbfile)
            except Exception:
                num_failed += 1
                failed_save.append((pocket_fn, ligand_fn))
                print(failed_save[-1])
                pbar.set_description(f"#failed: {num_failed}")
                continue

            try:
                if args.with_docking_scores:
                    if Path(pdbfile).stem not in recs:
                        print(
                            f"PDB {Path(pdbfile).stem} not found in list of docking files"
                        )
                        continue
                    else:
                        docking_scores.append(recs[Path(pdbfile).stem])

                ligand_data, pocket_data = process_ligand_and_pocket(
                    pdbfile,
                    sdffile,
                    dist_cutoff=args.dist_cutoff,
                    ca_only=args.ca_only,
                    no_H=args.no_H,
                    complex_creation_mode=args.complex_creation_mode,
                    com_weights=args.com_weights,
                )
            except (
                KeyError,
                AssertionError,
                FileNotFoundError,
                IndexError,
                ValueError,
            ) as e:
                print(type(e).__name__, e, pocket_fn, ligand_fn)
                num_failed += 1
                pbar.set_description(f"#failed: {num_failed}")
                continue

            pocket_name = ("-").join(pocket_fn.split("/")[1].split("_"))
            ligand_name = ("-").join(ligand_fn.split("/")[1].split("_"))
            pdb_and_mol_ids.append(f"{pocket_name}_{ligand_name}")
            lig_coords.append(ligand_data["lig_coords"])
            lig_mask.append(count * np.ones(len(ligand_data["lig_coords"])))
            lig_atom.append(ligand_data["lig_atoms"])
            lig_mol.append(ligand_data["lig_mol"])
            pocket_coords.append(pocket_data["pocket_coords"])
            pocket_atom.append(pocket_data["pocket_atoms"])
            pocket_atom_names.append(pocket_data["pocket_atom_names"])
            pocket_mask.append(count * np.ones(len(pocket_data["pocket_coords"])))
            pocket_resids.append(pocket_data["pocket_resids"])
            pocket_chainids.append(pocket_data["pocket_chainids"])
            pocket_resnames.append(pocket_data["pocket_resnames"])
            # new
            if not args.ca_only:
                pocket_one_hot_resids.append(pocket_data["pocket_one_hot"])
                pocket_ca_mask.append(pocket_data["pocket_ca_mask"])

            count_protein.append(pocket_data["pocket_coords"].shape[0])
            count_ligand.append(ligand_data["lig_coords"].shape[0])
            count_total.append(
                pocket_data["pocket_coords"].shape[0]
                + ligand_data["lig_coords"].shape[0]
            )
            count += 1

            # if split in {"val", "test"}:
            # Copy PDB file
            new_rec_name = Path(pdbfile).stem.replace("_", "-")
            pdb_file_out = Path(pdb_sdf_dir, f"{new_rec_name}.pdb")
            shutil.copy(pdbfile, pdb_file_out)

            # Copy SDF file
            new_lig_name = new_rec_name + "_" + Path(sdffile).stem.replace("_", "-")
            sdf_file_out = Path(pdb_sdf_dir, f"{new_lig_name}.sdf")
            shutil.copy(sdffile, sdf_file_out)

            # specify pocket residues
            with open(Path(pdb_sdf_dir, f"{new_lig_name}.txt"), "w") as f:
                f.write(" ".join(pocket_data["pocket_chainids"]))

        lig_coords = np.concatenate(lig_coords, axis=0)
        lig_atom = np.concatenate(lig_atom, axis=0)
        lig_mask = np.concatenate(lig_mask, axis=0)
        lig_mol = np.array(lig_mol)
        pocket_coords = np.concatenate(pocket_coords, axis=0)
        pocket_atom = np.concatenate(pocket_atom, axis=0)
        pocket_atom_names = np.concatenate(pocket_atom_names, axis=0)
        pocket_mask = np.concatenate(pocket_mask, axis=0)
        pocket_resids = np.concatenate(pocket_resids, axis=0)
        pocket_chainids = np.concatenate(pocket_chainids, axis=0)
        pocket_resnames = np.concatenate(pocket_resnames, axis=0)
        docking_scores = np.array(docking_scores)

        if not args.ca_only:
            pocket_one_hot_resids = np.concatenate(pocket_one_hot_resids, axis=0)
            pocket_ca_mask = np.concatenate(pocket_ca_mask, axis=0)
        else:
            pocket_one_hot_resids = np.array([])
            pocket_ca_mask = np.array([])

        d = "_dock" if args.with_docking_scores else ""
        saveall(
            processed_dir / f"{split}{d}.npz",
            pdb_and_mol_ids,
            lig_coords,
            lig_atom,
            lig_mask,
            lig_mol,
            pocket_coords,
            pocket_atom,
            pocket_atom_names,
            pocket_mask,
            pocket_resids,
            pocket_chainids,
            pocket_resnames,
            pocket_one_hot=pocket_one_hot_resids,
            pocket_ca_mask=pocket_ca_mask,
            docking_scores=docking_scores,
        )

        n_samples_after[split] = len(pdb_and_mol_ids)
        print(f"Processing {split} set took {(time() - tic)/60.0:.2f} minutes")

    # --------------------------------------------------------------------------
    # Compute statistics & additional information
    # --------------------------------------------------------------------------
    with np.load(processed_dir / f"train{d}.npz", allow_pickle=True) as data:
        lig_mask = data["lig_mask"]
        pocket_mask = data["pocket_mask"]
        lig_coords = data["lig_coords"]
        lig_atom = data["lig_atom"]
        pocket_atom = data["pocket_atom"]

    # Compute SMILES for all training examples
    train_smiles = compute_smiles(lig_coords, lig_atom, lig_mask)
    np.save(processed_dir / "train_smiles.npy", train_smiles)

    # Joint histogram of number of ligand and pocket nodes
    n_nodes = get_n_nodes(lig_mask, pocket_mask, smooth_sigma=1.0)
    np.save(Path(processed_dir, "size_distribution.npy"), n_nodes)

    # Convert bond length dictionaries to arrays for batch processing
    bonds1, bonds2, bonds3 = get_bond_length_arrays(atom_dict)

    # Get bond length definitions for Lennard-Jones potential
    rm_LJ = get_lennard_jones_rm(atom_dict)

    # Get histograms of ligand and pocket node types
    atom_hist, aa_hist = get_type_histograms(
        lig_atom, pocket_atom, atom_dict, aa_atom_encoder
    )

    # Create summary string
    summary_string = "# SUMMARY\n\n"
    summary_string += "# Before processing\n"
    summary_string += f"num_samples train: {n_train_before}\n"
    summary_string += f"num_samples val: {n_val_before}\n"
    summary_string += f"num_samples test: {n_test_before}\n\n"
    summary_string += "# After processing\n"
    summary_string += f"num_samples train: {n_samples_after['train']}\n"
    summary_string += f"num_samples val: {n_samples_after['val']}\n"
    summary_string += f"num_samples test: {n_samples_after['test']}\n\n"
    summary_string += "# Info\n"
    summary_string += f"'atom_encoder': {atom_dict}\n"
    summary_string += f"'atom_decoder': {list(atom_dict.keys())}\n"
    summary_string += f"'aa_encoder': {amino_acid_dict}\n"
    summary_string += f"'aa_decoder': {list(amino_acid_dict.keys())}\n"
    summary_string += f"'aa_atom_encoder': {aa_atom_encoder}\n"
    summary_string += f"'aa_atom_decoder': {list(aa_atom_encoder.keys())}\n"
    summary_string += f"'bonds1': {bonds1.tolist()}\n"
    summary_string += f"'bonds2': {bonds2.tolist()}\n"
    summary_string += f"'bonds3': {bonds3.tolist()}\n"
    summary_string += f"'lennard_jones_rm': {rm_LJ.tolist()}\n"
    summary_string += f"'atom_hist': {atom_hist}\n"
    summary_string += f"'aa_hist': {aa_hist}\n"
    summary_string += f"'n_nodes': {n_nodes.tolist()}\n"

    sns.distplot(count_protein)
    plt.savefig(processed_dir / "protein_size_distribution.png")
    plt.clf()

    sns.distplot(count_ligand)
    plt.savefig(processed_dir / "lig_size_distribution.png")
    plt.clf()

    sns.distplot(count_total)
    plt.savefig(processed_dir / "total_size_distribution.png")
    plt.clf()

    # Write summary to text file
    with open(processed_dir / "summary.txt", "w") as f:
        f.write(summary_string)

    # Print summary
    print(summary_string)

    print(failed_save)
