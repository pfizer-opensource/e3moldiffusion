import torch
from rdkit import Chem
import numpy as np

from rmsd import kabsch_rmsd
from experiments.xtb_wrapper import xtb_calculate
import os
import concurrent.futures


def get_bond_lengths(adj_matrix, coords):
    b_lengths = []
    for i in range(adj_matrix.shape[0]):
        for j in range(adj_matrix.shape[0]):
            if i >= j:
                if adj_matrix[j, i] > 0:
                    b_lengths.append(np.sqrt(((coords[i] - coords[j]) ** 2).sum()))
    return b_lengths


def get_bond_angles(adj_matrix, coords):
    vecs = torch.tensor(coords) - torch.tensor(coords).unsqueeze(1)
    # Remove non bonded
    vecs[~torch.tensor(adj_matrix).bool()] = torch.nan
    # Calculate angles from dot product
    vecs /= torch.norm(vecs, p=2, dim=-1, keepdim=True)
    costheta = torch.einsum("bij,bkj->bik", vecs, vecs)
    angles = torch.arccos(costheta) * 180 / torch.pi
    # Mask out diagonals, self angles and non bonded
    mask_diagonal = torch.stack([~torch.eye(angles.shape[0]).bool()] * angles.shape[0])
    mask_notnan = ~angles.isnan()
    relevant_angles = angles[mask_diagonal & mask_notnan]
    return relevant_angles


def enumerateTorsions(mol):
    torsionSmarts = "[!$(*#*)&!D1]~[!$(*#*)&!D1]"
    torsionQuery = Chem.MolFromSmarts(torsionSmarts)
    matches = mol.GetSubstructMatches(torsionQuery)
    torsionList = []
    for match in matches:
        idx2 = match[0]
        idx3 = match[1]
        bond = mol.GetBondBetweenAtoms(idx2, idx3)
        jAtom = mol.GetAtomWithIdx(idx2)
        kAtom = mol.GetAtomWithIdx(idx3)
        if (
            (jAtom.GetHybridization() != Chem.HybridizationType.SP2)
            and (jAtom.GetHybridization() != Chem.HybridizationType.SP3)
        ) or (
            (kAtom.GetHybridization() != Chem.HybridizationType.SP2)
            and (kAtom.GetHybridization() != Chem.HybridizationType.SP3)
        ):
            continue
        for b1 in jAtom.GetBonds():
            if b1.GetIdx() == bond.GetIdx():
                continue
            idx1 = b1.GetOtherAtomIdx(idx2)
            for b2 in kAtom.GetBonds():
                if (b2.GetIdx() == bond.GetIdx()) or (b2.GetIdx() == b1.GetIdx()):
                    continue
                idx4 = b2.GetOtherAtomIdx(idx3)
                # skip 3-membered rings
                if idx4 == idx1:
                    continue
                torsionList.append((idx1, idx2, idx3, idx4))
    return torsionList


def get_dihedral_angles(torsion_list, coords):
    t_angles = []
    for t in torsion_list:
        u1, u2, u3, u4 = coords[torch.tensor(t)]

        a1 = u2 - u1
        a2 = u3 - u2
        a3 = u4 - u3

        v1 = np.cross(a1, a2)
        v1 = v1 / (v1 * v1).sum(-1) ** 0.5
        v2 = np.cross(a2, a3)
        v2 = v2 / (v2 * v2).sum(-1) ** 0.5
        porm = np.sign((v1 * a3).sum(-1))
        rad = np.arccos(
            (v1 * v2).sum(-1) / ((v1**2).sum(-1) * (v2**2).sum(-1)) ** 0.5
        )
        if not porm == 0:
            rad = rad * porm
        t_angles.append(rad * 180 / torch.pi)
    return t_angles


def change_internal_coordinates(
    mol,
    save_xyz_idx=None,
    xtb_options={"opt": True},
    energy_diff=True,
):
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    coords = mol.GetConformer().GetPositions()
    adj_matrix = Chem.rdmolops.GetAdjacencyMatrix(mol)
    torsions = enumerateTorsions(mol)

    results = xtb_calculate(
        atoms, coords, charge=Chem.GetFormalCharge(mol), options=xtb_options, n_cores=1
    )
    if not results["normal_termination"]:
        return [], [], [], np.nan, np.nan

    if energy_diff:
        sp_options = {i: xtb_options[i] for i in xtb_options if i != "opt"}
        results_sp = xtb_calculate(
            atoms,
            coords,
            charge=Chem.GetFormalCharge(mol),
            options=sp_options,
            n_cores=1,
        )
    opt_coords = np.array(results["opt_coords"])

    b_lengths = np.array(get_bond_lengths(adj_matrix, coords))
    b_lengths_opt = np.array(get_bond_lengths(adj_matrix, opt_coords))
    diff_b_lengths = b_lengths - b_lengths_opt

    b_angles = np.array(get_bond_angles(adj_matrix, coords))
    b_angles_opt = np.array(get_bond_angles(adj_matrix, opt_coords))
    diff_b_angles = b_angles - b_angles_opt

    d_angles = np.array(get_dihedral_angles(torsions, coords))
    d_angles_opt = np.array(get_dihedral_angles(torsions, opt_coords))
    diff_d_angles = d_angles - d_angles_opt

    rmsd = kabsch_rmsd(coords, opt_coords, translate=True)
    if energy_diff:
        diff_e = results_sp["total_energy"] - results["total_energy"]
    else:
        diff_e = np.nan

    if save_xyz_idx:
        xyz = f"{len(atoms)}\n\n"
        for atom, coord in zip(atoms, coords):
            xyz += f"{atom} {coord[0]:.8f} {coord[1]:.8f} {coord[2]:.8f}\n"
        with open(f"../mol_{save_xyz_idx:03d}.xyz", "w") as f:
            f.write(xyz)

        xyz_opt = f"{len(atoms)}\n\n"
        for atom, coord in zip(atoms, opt_coords):
            xyz_opt += f"{atom} {coord[0]:.8f} {coord[1]:.8f} {coord[2]:.8f}\n"
        with open(f"../mol_opt_{save_xyz_idx:03d}.xyz", "w") as f:
            f.write(xyz_opt)

    return diff_b_lengths, diff_b_angles, diff_d_angles, rmsd, diff_e


def calc_diff_mols(mols, n_cores=1, save_xyz_coords=False):
    ids = np.arange(len(mols)) + 1 if save_xyz_coords else [None] * len(mols)
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_cores) as executor:
        results = executor.map(change_internal_coordinates, mols, ids)
    results = [res for res in results]

    diff_b_lengths = []
    diff_b_angles = []
    diff_d_angles = []
    rmsds = []
    diff_es = []

    for res in results:
        diff_b_lengths.extend(list(res[0]))
        diff_b_angles.extend(list(res[1]))
        diff_d_angles.extend(list(res[2]))
        rmsds.append(res[3])
        diff_es.append(res[4])

    return diff_b_lengths, diff_b_angles, diff_d_angles, rmsds, diff_es


if __name__ == "__main__":
    N_CORES = 8
    SAVE_XYZ_FILES = False

    import sys
    import pickle
    import matplotlib.pyplot as plt
    from pathlib import Path

    filename = str(sys.argv[-1])
    print(
        f"Calculating change of internal coordinates for molecules in file {filename}"
    )

    with open(filename, "rb") as f:
        data = pickle.load(f)

    mols = [d.rdkit_mol for d in data]
    for i, mol in enumerate(mols):
        try:
            Chem.SanitizeMol(mol)
        except Exception as e:
            print(f"Skipping molecule {i} because of sanitization error: {e}")

    diff_b_lengths, diff_b_angles, diff_d_angles, rmsds, diff_es = calc_diff_mols(
        mols, N_CORES, SAVE_XYZ_FILES
    )

    save_dir = os.path.dirname(filename)

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
