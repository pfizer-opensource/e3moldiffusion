from pathlib import Path
from time import time
import random
from collections import defaultdict
import argparse
import tempfile
import warnings
from tqdm import tqdm
import numpy as np
import torch
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import three_to_one, is_aa
from Bio.PDB import PDBIO
from openbabel import openbabel
from rdkit import Chem
from rdkit.Chem import QED
from scipy.ndimage import gaussian_filter
import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem.rdForceFieldHelpers import UFFOptimizeMolecule, UFFHasAllMoleculeParams
import openbabel


def get_bond_order(atom1, atom2, distance):
    distance = 100 * distance  # We change the metric

    if (
        atom1 in bonds3
        and atom2 in bonds3[atom1]
        and distance < bonds3[atom1][atom2] + margin3
    ):
        return 3  # Triple

    if (
        atom1 in bonds2
        and atom2 in bonds2[atom1]
        and distance < bonds2[atom1][atom2] + margin2
    ):
        return 2  # Double

    if (
        atom1 in bonds1
        and atom2 in bonds1[atom1]
        and distance < bonds1[atom1][atom2] + margin1
    ):
        return 1  # Single

    return 0  # No bond


def get_bond_order_batch(atoms1, atoms2, distances, dataset_info):
    if isinstance(atoms1, np.ndarray):
        atoms1 = torch.from_numpy(atoms1)
    if isinstance(atoms2, np.ndarray):
        atoms2 = torch.from_numpy(atoms2)
    if isinstance(distances, np.ndarray):
        distances = torch.from_numpy(distances)

    distances = 100 * distances  # We change the metric

    bonds1 = torch.tensor(dataset_info["bonds1"], device=atoms1.device)
    bonds2 = torch.tensor(dataset_info["bonds2"], device=atoms1.device)
    bonds3 = torch.tensor(dataset_info["bonds3"], device=atoms1.device)

    bond_types = torch.zeros_like(atoms1)  # 0: No bond

    # Single
    bond_types[distances < bonds1[atoms1, atoms2] + margin1] = 1

    # Double (note that already assigned single bonds will be overwritten)
    bond_types[distances < bonds2[atoms1, atoms2] + margin2] = 2

    # Triple
    bond_types[distances < bonds3[atoms1, atoms2] + margin3] = 3

    return bond_types


def make_mol_openbabel(positions, atom_types, atom_decoder):
    """
    Build an RDKit molecule using openbabel for creating bonds
    Args:
        positions: N x 3
        atom_types: N
        atom_decoder: maps indices to atom types
    Returns:
        rdkit molecule
    """
    atom_types = [atom_decoder[x] for x in atom_types]

    with tempfile.NamedTemporaryFile() as tmp:
        tmp_file = tmp.name

        # Write xyz file
        write_xyz_file(positions, atom_types, tmp_file)

        # Convert to sdf file with openbabel
        # openbabel will add bonds
        obConversion = openbabel.OBConversion()
        obConversion.SetInAndOutFormats("xyz", "sdf")
        ob_mol = openbabel.OBMol()
        obConversion.ReadFile(ob_mol, tmp_file)

        obConversion.WriteFile(ob_mol, tmp_file)

        # Read sdf file with RDKit
        mol = Chem.SDMolSupplier(tmp_file, sanitize=False)[0]

    return mol


def make_mol_edm(positions, atom_types, dataset_info, add_coords):
    """
    Equivalent to EDM's way of building RDKit molecules
    """
    n = len(positions)

    # (X, A, E): atom_types, adjacency matrix, edge_types
    # X: N (int)
    # A: N x N (bool) -> (binary adjacency matrix)
    # E: N x N (int) -> (bond type, 0 if no bond)
    pos = positions.unsqueeze(0)  # add batch dim
    dists = torch.cdist(pos, pos, p=2).squeeze(0).view(-1)  # remove batch dim & flatten
    atoms1, atoms2 = torch.cartesian_prod(atom_types, atom_types).T
    E_full = get_bond_order_batch(atoms1, atoms2, dists, dataset_info).view(n, n)
    E = torch.tril(E_full, diagonal=-1)  # Warning: the graph should be DIRECTED
    A = E.bool()
    X = atom_types

    mol = Chem.RWMol()
    for atom in X:
        a = Chem.Atom(dataset_info["atom_decoder"][atom.item()])
        mol.AddAtom(a)

    all_bonds = torch.nonzero(A)
    for bond in all_bonds:
        mol.AddBond(
            bond[0].item(), bond[1].item(), bond_dict[E[bond[0], bond[1]].item()]
        )

    if add_coords:
        conf = Chem.Conformer(mol.GetNumAtoms())
        for i in range(mol.GetNumAtoms()):
            conf.SetAtomPosition(
                i,
                (
                    positions[i, 0].item(),
                    positions[i, 1].item(),
                    positions[i, 2].item(),
                ),
            )
        mol.AddConformer(conf)

    return mol


def build_molecule(
    positions, atom_types, dataset_info, add_coords=False, use_openbabel=True
):
    """
    Build RDKit molecule
    Args:
        positions: N x 3
        atom_types: N
        dataset_info: dict
        add_coords: Add conformer to mol (always added if use_openbabel=True)
        use_openbabel: use OpenBabel to create bonds
    Returns:
        RDKit molecule
    """
    if use_openbabel:
        mol = make_mol_openbabel(positions, atom_types, dataset_info["atom_decoder"])
    else:
        mol = make_mol_edm(positions, atom_types, dataset_info, add_coords)

    return mol


def process_molecule(
    rdmol, add_hydrogens=False, sanitize=False, relax_iter=0, largest_frag=False
):
    """
    Apply filters to an RDKit molecule. Makes a copy first.
    Args:
        rdmol: rdkit molecule
        add_hydrogens
        sanitize
        relax_iter: maximum number of UFF optimization iterations
        largest_frag: filter out the largest fragment in a set of disjoint
            molecules
    Returns:
        RDKit molecule or None if it does not pass the filters
    """

    # Create a copy
    mol = Chem.Mol(rdmol)

    if sanitize:
        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            warnings.warn("Sanitization failed. Returning None.")
            return None

    if add_hydrogens:
        mol = Chem.AddHs(mol, addCoords=(len(mol.GetConformers()) > 0))

    if largest_frag:
        mol_frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
        mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
        if sanitize:
            # sanitize the updated molecule
            try:
                Chem.SanitizeMol(mol)
            except ValueError:
                return None

    if relax_iter > 0:
        if not UFFHasAllMoleculeParams(mol):
            warnings.warn(
                "UFF parameters not available for all atoms. " "Returning None."
            )
            return None

        try:
            uff_relax(mol, relax_iter)
            if sanitize:
                # sanitize the updated molecule
                Chem.SanitizeMol(mol)
        except (RuntimeError, ValueError) as e:
            return None

    return mol


def uff_relax(mol, max_iter=200):
    """
    Uses RDKit's universal force field (UFF) implementation to optimize a
    molecule.
    """
    more_iterations_required = UFFOptimizeMolecule(mol, maxIters=max_iter)
    if more_iterations_required:
        warnings.warn(
            f"Maximum number of FF iterations reached. "
            f"Returning molecule after {max_iter} relaxation steps."
        )
    return more_iterations_required


def filter_rd_mol(rdmol):
    """
    Filter out RDMols if they have a 3-3 ring intersection
    adapted from:
    https://github.com/luost26/3D-Generative-SBDD/blob/main/utils/chem.py
    """
    ring_info = rdmol.GetRingInfo()
    ring_info.AtomRings()
    rings = [set(r) for r in ring_info.AtomRings()]

    # 3-3 ring intersection
    for i, ring_a in enumerate(rings):
        if len(ring_a) != 3:
            continue
        for j, ring_b in enumerate(rings):
            if i <= j:
                continue
            inter = ring_a.intersection(ring_b)
            if (len(ring_b) == 3) and (len(inter) > 0):
                return False

    return True


def rotation_matrix(angle, axis):
    """
    Args:
        angle: (n,)
        axis: 0=x, 1=y, 2=z
    Returns:
        (n, 3, 3)
    """
    n = len(angle)
    R = np.eye(3)[None, :, :].repeat(n, axis=0)

    axis = 2 - axis
    start = axis // 2
    step = axis % 2 + 1
    s = slice(start, start + step + 1, step)

    R[:, s, s] = np.array(
        [
            [np.cos(angle), (-1) ** (axis + 1) * np.sin(angle)],
            [(-1) ** axis * np.sin(angle), np.cos(angle)],
        ]
    ).transpose(2, 0, 1)
    return R


def get_bb_transform(n_xyz, ca_xyz, c_xyz):
    """
    Compute translation and rotation of the canoncical backbone frame (triangle N-Ca-C) from a position with
    Ca at the origin, N on the x-axis and C in the xy-plane to the global position of the backbone frame

    Args:
        n_xyz: (n, 3)
        ca_xyz: (n, 3)
        c_xyz: (n, 3)

    Returns:
        quaternion represented as array of shape (n, 4)
        translation vector which is an array of shape (n, 3)
    """

    translation = ca_xyz
    n_xyz = n_xyz - translation
    c_xyz = c_xyz - translation

    # Find rotation matrix that aligns the coordinate systems
    #    rotate around y-axis to move N into the xy-plane
    theta_y = np.arctan2(n_xyz[:, 2], -n_xyz[:, 0])
    Ry = rotation_matrix(theta_y, 1)
    n_xyz = np.einsum("noi,ni->no", Ry.transpose(0, 2, 1), n_xyz)

    #    rotate around z-axis to move N onto the x-axis
    theta_z = np.arctan2(n_xyz[:, 1], n_xyz[:, 0])
    Rz = rotation_matrix(theta_z, 2)
    # n_xyz = np.einsum('noi,ni->no', Rz.transpose(0, 2, 1), n_xyz)

    #    rotate around x-axis to move C into the xy-plane
    c_xyz = np.einsum(
        "noj,nji,ni->no", Rz.transpose(0, 2, 1), Ry.transpose(0, 2, 1), c_xyz
    )
    theta_x = np.arctan2(c_xyz[:, 2], c_xyz[:, 1])
    Rx = rotation_matrix(theta_x, 0)

    # Final rotation matrix
    R = np.einsum("nok,nkj,nji->noi", Ry, Rz, Rx)

    # Convert to quaternion
    # q = w + i*u_x + j*u_y + k * u_z
    quaternion = rotation_matrix_to_quaternion(R)

    return quaternion, translation


def get_bb_coords_from_transform(ca_coords, quaternion):
    """
    Args:
        ca_coords: (n, 3)
        quaternion: (n, 4)
    Returns:
        backbone coords (n*3, 3), order is [N, CA, C]
        backbone atom types as a list of length n*3
    """
    R = quaternion_to_rotation_matrix(quaternion)
    bb_coords = np.tile(
        np.array(
            [
                [N_CA_DIST, 0, 0],
                [0, 0, 0],
                [CA_C_DIST * np.cos(N_CA_C_ANGLE), CA_C_DIST * np.sin(N_CA_C_ANGLE), 0],
            ]
        ),
        [len(ca_coords), 1],
    )
    bb_coords = np.einsum(
        "noi,ni->no", R.repeat(3, axis=0), bb_coords
    ) + ca_coords.repeat(3, axis=0)
    bb_atom_types = [t for _ in range(len(ca_coords)) for t in ["N", "C", "C"]]

    return bb_coords, bb_atom_types


def quaternion_to_rotation_matrix(q):
    """
    x_rot = R x

    Args:
        q: (n, 4)
    Returns:
        R: (n, 3, 3)
    """
    # Normalize
    q = q / (q**2).sum(1, keepdims=True) ** 0.5

    # https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.stack(
        [
            np.stack(
                [
                    1 - 2 * y**2 - 2 * z**2,
                    2 * x * y - 2 * z * w,
                    2 * x * z + 2 * y * w,
                ],
                axis=1,
            ),
            np.stack(
                [
                    2 * x * y + 2 * z * w,
                    1 - 2 * x**2 - 2 * z**2,
                    2 * y * z - 2 * x * w,
                ],
                axis=1,
            ),
            np.stack(
                [
                    2 * x * z - 2 * y * w,
                    2 * y * z + 2 * x * w,
                    1 - 2 * x**2 - 2 * y**2,
                ],
                axis=1,
            ),
        ],
        axis=1,
    )

    return R


def rotation_matrix_to_quaternion(R):
    """
    https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
    Args:
        R: (n, 3, 3)
    Returns:
        q: (n, 4)
    """

    t = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    r = np.sqrt(1 + t)
    w = 0.5 * r
    x = np.sign(R[:, 2, 1] - R[:, 1, 2]) * np.abs(
        0.5 * np.sqrt(1 + R[:, 0, 0] - R[:, 1, 1] - R[:, 2, 2])
    )
    y = np.sign(R[:, 0, 2] - R[:, 2, 0]) * np.abs(
        0.5 * np.sqrt(1 - R[:, 0, 0] + R[:, 1, 1] - R[:, 2, 2])
    )
    z = np.sign(R[:, 1, 0] - R[:, 0, 1]) * np.abs(
        0.5 * np.sqrt(1 - R[:, 0, 0] - R[:, 1, 1] + R[:, 2, 2])
    )

    return np.stack((w, x, y, z), axis=1)


def read_label_file(csv_path):
    """
    Read BindingMOAD's label file
    Args:
        csv_path: path to 'every.csv'
    Returns:
        Nested dictionary with all ligands. First level: EC number,
            Second level: PDB ID, Third level: list of ligands. Each ligand is
            represented as a tuple (ligand name, validity, SMILES string)
    """
    ligand_dict = {}

    with open(csv_path, "r") as f:
        for line in f.readlines():
            row = line.split(",")

            # new protein class
            if len(row[0]) > 0:
                curr_class = row[0]
                ligand_dict[curr_class] = {}
                continue

            # new protein
            if len(row[2]) > 0:
                curr_prot = row[2]
                ligand_dict[curr_class][curr_prot] = []
                continue

            # new small molecule
            if len(row[3]) > 0:
                ligand_dict[curr_class][curr_prot].append(
                    # (ligand name, validity, SMILES string)
                    [row[3], row[4], row[9]]
                )

    return ligand_dict


def compute_druglikeness(ligand_dict):
    """
    Computes RDKit's QED value and adds it to the dictionary
    Args:
        ligand_dict: nested ligand dictionary
    Returns:
        the same ligand dictionary with additional QED values
    """
    print("Computing QED values...")
    for p, m in tqdm(
        [(p, m) for c in ligand_dict for p in ligand_dict[c] for m in ligand_dict[c][p]]
    ):
        mol = Chem.MolFromSmiles(m[2])
        if mol is None:
            mol_id = f"{p}_{m}"
            warnings.warn(
                f"Could not construct molecule {mol_id} from SMILES " f"string '{m[2]}'"
            )
            continue
        m.append(QED.qed(mol))
    return ligand_dict


def filter_and_flatten(ligand_dict, qed_thresh, max_occurences, seed):
    filtered_examples = []
    all_examples = [
        (c, p, m)
        for c in ligand_dict
        for p in ligand_dict[c]
        for m in ligand_dict[c][p]
    ]

    # shuffle to select random examples of ligands that occur more than
    # max_occurences times
    random.seed(seed)
    random.shuffle(all_examples)

    ligand_name_counter = defaultdict(int)
    print("Filtering examples...")
    for c, p, m in tqdm(all_examples):
        ligand_name, ligand_chain, ligand_resi = m[0].split(":")
        if m[1] == "valid" and len(m) > 3 and m[3] > qed_thresh:
            if ligand_name_counter[ligand_name] < max_occurences:
                filtered_examples.append((c, p, m))
                ligand_name_counter[ligand_name] += 1

    return filtered_examples


def split_by_ec_number(data_list, n_val, n_test, ec_level=1):
    """
    Split dataset into training, validation and test sets based on EC numbers
    https://en.wikipedia.org/wiki/Enzyme_Commission_number
    Args:
        data_list: list of ligands
        n_val: number of validation examples
        n_test: number of test examples
        ec_level: level in the EC numbering hierarchy at which the split is
            made, i.e. items with matching EC numbers at this level are put in
            the same set
    Returns:
        dictionary with keys 'train', 'val', and 'test'
    """

    examples_per_class = defaultdict(int)
    for c, p, m in data_list:
        c_sub = ".".join(c.split(".")[:ec_level])
        examples_per_class[c_sub] += 1

    assert sum(examples_per_class.values()) == len(data_list)

    # split ec numbers
    val_classes = set()
    for c, num in sorted(examples_per_class.items(), key=lambda x: x[1], reverse=True):
        if sum([examples_per_class[x] for x in val_classes]) + num <= n_val:
            val_classes.add(c)

    test_classes = set()
    for c, num in sorted(examples_per_class.items(), key=lambda x: x[1], reverse=True):
        # skip classes already used in the validation set
        if c in val_classes:
            continue
        if sum([examples_per_class[x] for x in test_classes]) + num <= n_test:
            test_classes.add(c)

    # remaining classes belong to test set
    train_classes = {
        x for x in examples_per_class if x not in val_classes and x not in test_classes
    }

    # create separate lists of examples
    data_split = {}
    data_split["train"] = [
        x for x in data_list if ".".join(x[0].split(".")[:ec_level]) in train_classes
    ]
    data_split["val"] = [
        x for x in data_list if ".".join(x[0].split(".")[:ec_level]) in val_classes
    ]
    data_split["test"] = [
        x for x in data_list if ".".join(x[0].split(".")[:ec_level]) in test_classes
    ]

    assert len(data_split["train"]) + len(data_split["val"]) + len(
        data_split["test"]
    ) == len(data_list)

    return data_split


def ligand_list_to_dict(ligand_list):
    out_dict = defaultdict(list)
    for _, p, m in ligand_list:
        out_dict[p].append(m)
    return out_dict


def process_ligand_and_pocket(
    pdb_struct,
    ligand_name,
    ligand_chain,
    ligand_resi,
    dist_cutoff,
    ca_only,
    compute_quaternion=False,
):
    try:
        residues = {
            obj.id[1]: obj for obj in pdb_struct[0][ligand_chain].get_residues()
        }
    except KeyError as e:
        raise KeyError(
            f"Chain {e} not found ({pdbfile}, "
            f"{ligand_name}:{ligand_chain}:{ligand_resi})"
        )
    ligand = residues[ligand_resi]
    assert (
        ligand.get_resname() == ligand_name
    ), f"{ligand.get_resname()} != {ligand_name}"

    # remove H atoms if not in atom_dict, other atom types that aren't allowed
    # should stay so that the entire ligand can be removed from the dataset
    lig_atoms = [
        a
        for a in ligand.get_atoms()
        if (a.element.capitalize() in atom_dict or a.element != "H")
    ]
    lig_coords = np.array([a.get_coord() for a in lig_atoms])

    try:
        lig_one_hot = np.stack(
            [
                np.eye(1, len(atom_dict), atom_dict[a.element.capitalize()]).squeeze()
                for a in lig_atoms
            ]
        )
    except KeyError as e:
        raise KeyError(
            f"Ligand atom {e} not in atom dict ({pdbfile}, "
            f"{ligand_name}:{ligand_chain}:{ligand_resi})"
        )

    # Find interacting pocket residues based on distance cutoff
    pocket_residues = []
    for residue in pdb_struct[0].get_residues():
        res_coords = np.array([a.get_coord() for a in residue.get_atoms()])
        if (
            is_aa(residue.get_resname(), standard=True)
            and (
                ((res_coords[:, None, :] - lig_coords[None, :, :]) ** 2).sum(-1) ** 0.5
            ).min()
            < dist_cutoff
        ):
            pocket_residues.append(residue)

    # Compute transform of the canonical reference frame
    n_xyz = np.array([res["N"].get_coord() for res in pocket_residues])
    ca_xyz = np.array([res["CA"].get_coord() for res in pocket_residues])
    c_xyz = np.array([res["C"].get_coord() for res in pocket_residues])

    if compute_quaternion:
        quaternion, c_alpha = get_bb_transform(n_xyz, ca_xyz, c_xyz)
        if np.any(np.isnan(quaternion)):
            raise ValueError(
                f"Invalid value in quaternion ({pdbfile}, "
                f"{ligand_name}:{ligand_chain}:{ligand_resi})"
            )
    else:
        c_alpha = ca_xyz

    if ca_only:
        pocket_coords = c_alpha
        try:
            pocket_one_hot = np.stack(
                [
                    np.eye(
                        1,
                        len(amino_acid_dict),
                        amino_acid_dict[three_to_one(res.get_resname())],
                    ).squeeze()
                    for res in pocket_residues
                ]
            )
        except KeyError as e:
            raise KeyError(
                f"{e} not in amino acid dict ({pdbfile}, "
                f"{ligand_name}:{ligand_chain}:{ligand_resi})"
            )
    else:
        pocket_atoms = [
            a
            for res in pocket_residues
            for a in res.get_atoms()
            if (a.element.capitalize() in atom_dict or a.element != "H")
        ]
        pocket_coords = np.array([a.get_coord() for a in pocket_atoms])
        try:
            pocket_one_hot = np.stack(
                [
                    np.eye(
                        1, len(atom_dict), atom_dict[a.element.capitalize()]
                    ).squeeze()
                    for a in pocket_atoms
                ]
            )
        except KeyError as e:
            raise KeyError(
                f"Pocket atom {e} not in atom dict ({pdbfile}, "
                f"{ligand_name}:{ligand_chain}:{ligand_resi})"
            )

    pocket_ids = [f"{res.parent.id}:{res.id[1]}" for res in pocket_residues]

    ligand_data = {
        "lig_coords": lig_coords,
        "lig_one_hot": lig_one_hot,
    }
    pocket_data = {
        "pocket_ca": pocket_coords,
        "pocket_one_hot": pocket_one_hot,
        "pocket_ids": pocket_ids,
    }
    if compute_quaternion:
        pocket_data["pocket_quaternion"] = quaternion
    return ligand_data, pocket_data


def compute_smiles(positions, one_hot, mask):
    print("Computing SMILES ...")

    atom_types = np.argmax(one_hot, axis=-1)

    sections = np.where(np.diff(mask))[0] + 1
    positions = [torch.from_numpy(x) for x in np.split(positions, sections)]
    atom_types = [torch.from_numpy(x) for x in np.split(atom_types, sections)]

    mols_smiles = []

    pbar = tqdm(enumerate(zip(positions, atom_types)), total=len(np.unique(mask)))
    for i, (pos, atom_type) in pbar:
        mol = build_molecule(pos, atom_type, dataset_info)

        # BasicMolecularMetrics() computes SMILES after sanitization
        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            continue

        mol = Chem.MolToSmiles(mol)
        if mol is not None:
            mols_smiles.append(mol)
        pbar.set_description(f"{len(mols_smiles)}/{i + 1} successful")

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
                # Replace missing values with sum of average covalent radii
                bond_len = covalent_radii[a1] + covalent_radii[a2]

            LJ_rm[atom_mapping[a1], atom_mapping[a2]] = bond_len

    assert np.all(LJ_rm == LJ_rm.T)
    return LJ_rm


def get_type_histograms(lig_one_hot, pocket_one_hot, atom_encoder, aa_encoder):
    atom_decoder = list(atom_encoder.keys())
    atom_counts = {k: 0 for k in atom_encoder.keys()}
    for a in [atom_decoder[x] for x in lig_one_hot.argmax(1)]:
        atom_counts[a] += 1

    aa_decoder = list(aa_encoder.keys())
    aa_counts = {k: 0 for k in aa_encoder.keys()}
    for r in [aa_decoder[x] for x in pocket_one_hot.argmax(1)]:
        aa_counts[r] += 1

    return atom_counts, aa_counts


def saveall(
    filename,
    pdb_and_mol_ids,
    lig_coords,
    lig_one_hot,
    lig_mask,
    pocket_c_alpha,
    pocket_quaternion,
    pocket_one_hot,
    pocket_mask,
):
    np.savez(
        filename,
        names=pdb_and_mol_ids,
        lig_coords=lig_coords,
        lig_one_hot=lig_one_hot,
        lig_mask=lig_mask,
        pocket_c_alpha=pocket_c_alpha,
        pocket_quaternion=pocket_quaternion,
        pocket_one_hot=pocket_one_hot,
        pocket_mask=pocket_mask,
    )
    return True


# ------------------------------------------------------------------------------
# Computational
# ------------------------------------------------------------------------------
FLOAT_TYPE = torch.float32
INT_TYPE = torch.int64


# ------------------------------------------------------------------------------
# Bond parameters
# ------------------------------------------------------------------------------

# margin1, margin2, margin3 = 10, 5, 3
margin1, margin2, margin3 = 3, 2, 1

allowed_bonds = {
    "H": 1,
    "C": 4,
    "N": 3,
    "O": 2,
    "F": 1,
    "B": 3,
    "Al": 3,
    "Si": 4,
    "P": [3, 5],
    "S": 4,
    "Cl": 1,
    "As": 3,
    "Br": 1,
    "I": 1,
    "Hg": [1, 2],
    "Bi": [3, 5],
}

# Bond lengths from:
# http://www.wiredchemist.com/chemistry/data/bond_energies_lengths.html
# And:
# http://chemistry-reference.com/tables/Bond%20Lengths%20and%20Enthalpies.pdf
bonds1 = {
    "H": {
        "H": 74,
        "C": 109,
        "N": 101,
        "O": 96,
        "F": 92,
        "B": 119,
        "Si": 148,
        "P": 144,
        "As": 152,
        "S": 134,
        "Cl": 127,
        "Br": 141,
        "I": 161,
    },
    "C": {
        "H": 109,
        "C": 154,
        "N": 147,
        "O": 143,
        "F": 135,
        "Si": 185,
        "P": 184,
        "S": 182,
        "Cl": 177,
        "Br": 194,
        "I": 214,
    },
    "N": {
        "H": 101,
        "C": 147,
        "N": 145,
        "O": 140,
        "F": 136,
        "Cl": 175,
        "Br": 214,
        "S": 168,
        "I": 222,
        "P": 177,
    },
    "O": {
        "H": 96,
        "C": 143,
        "N": 140,
        "O": 148,
        "F": 142,
        "Br": 172,
        "S": 151,
        "P": 163,
        "Si": 163,
        "Cl": 164,
        "I": 194,
    },
    "F": {
        "H": 92,
        "C": 135,
        "N": 136,
        "O": 142,
        "F": 142,
        "S": 158,
        "Si": 160,
        "Cl": 166,
        "Br": 178,
        "P": 156,
        "I": 187,
    },
    "B": {"H": 119, "Cl": 175},
    "Si": {
        "Si": 233,
        "H": 148,
        "C": 185,
        "O": 163,
        "S": 200,
        "F": 160,
        "Cl": 202,
        "Br": 215,
        "I": 243,
    },
    "Cl": {
        "Cl": 199,
        "H": 127,
        "C": 177,
        "N": 175,
        "O": 164,
        "P": 203,
        "S": 207,
        "B": 175,
        "Si": 202,
        "F": 166,
        "Br": 214,
    },
    "S": {
        "H": 134,
        "C": 182,
        "N": 168,
        "O": 151,
        "S": 204,
        "F": 158,
        "Cl": 207,
        "Br": 225,
        "Si": 200,
        "P": 210,
        "I": 234,
    },
    "Br": {
        "Br": 228,
        "H": 141,
        "C": 194,
        "O": 172,
        "N": 214,
        "Si": 215,
        "S": 225,
        "F": 178,
        "Cl": 214,
        "P": 222,
    },
    "P": {
        "P": 221,
        "H": 144,
        "C": 184,
        "O": 163,
        "Cl": 203,
        "S": 210,
        "F": 156,
        "N": 177,
        "Br": 222,
    },
    "I": {
        "H": 161,
        "C": 214,
        "Si": 243,
        "N": 222,
        "O": 194,
        "S": 234,
        "F": 187,
        "I": 266,
    },
    "As": {"H": 152},
}

bonds2 = {
    "C": {"C": 134, "N": 129, "O": 120, "S": 160},
    "N": {"C": 129, "N": 125, "O": 121},
    "O": {"C": 120, "N": 121, "O": 121, "P": 150},
    "P": {"O": 150, "S": 186},
    "S": {"P": 186, "C": 160},
}


bonds3 = {
    "C": {"C": 120, "N": 116, "O": 113},
    "N": {"C": 116, "N": 110},
    "O": {"C": 113},
}

bond_dict = [
    None,
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]

# https://en.wikipedia.org/wiki/Covalent_radius#Radii_for_multiple_bonds
# (2022/08/14)
covalent_radii = {
    "H": 32,
    "C": 60,
    "N": 54,
    "O": 53,
    "F": 53,
    "B": 73,
    "Al": 111,
    "Si": 102,
    "P": 94,
    "S": 94,
    "Cl": 93,
    "As": 106,
    "Br": 109,
    "I": 125,
    "Hg": 133,
    "Bi": 135,
}

# ------------------------------------------------------------------------------
# Backbone geometry
# Taken from: Bhagavan, N. V., and C. E. Ha.
# "Chapter 4-Three-dimensional structure of proteins and disorders of protein misfolding."
# Essentials of Medical Biochemistry (2015): 31-51.
# https://www.sciencedirect.com/science/article/pii/B978012416687500004X
# ------------------------------------------------------------------------------
N_CA_DIST = 1.47
CA_C_DIST = 1.53
N_CA_C_ANGLE = 110 * np.pi / 180


# ------------------------------------------------------------------------------
# Dataset-specific constants
# ------------------------------------------------------------------------------
dataset_params = {}
dataset_params["bindingmoad"] = {
    "atom_encoder": {
        "C": 0,
        "N": 1,
        "O": 2,
        "S": 3,
        "B": 4,
        "Br": 5,
        "Cl": 6,
        "P": 7,
        "I": 8,
        "F": 9,
    },
    "atom_decoder": ["C", "N", "O", "S", "B", "Br", "Cl", "P", "I", "F"],
    "aa_encoder": {
        "A": 0,
        "C": 1,
        "D": 2,
        "E": 3,
        "F": 4,
        "G": 5,
        "H": 6,
        "I": 7,
        "K": 8,
        "L": 9,
        "M": 10,
        "N": 11,
        "P": 12,
        "Q": 13,
        "R": 14,
        "S": 15,
        "T": 16,
        "V": 17,
        "W": 18,
        "Y": 19,
    },
    "aa_decoder": [
        "A",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "I",
        "K",
        "L",
        "M",
        "N",
        "P",
        "Q",
        "R",
        "S",
        "T",
        "V",
        "W",
        "Y",
    ],
    # PyMOL colors, see: https://pymolwiki.org/index.php/Color_Values#Chemical_element_colours
    "colors_dic": [
        "#33ff33",
        "#3333ff",
        "#ff4d4d",
        "#e6c540",
        "#ffb5b5",
        "#A62929",
        "#1FF01F",
        "#ff8000",
        "#940094",
        "#B3FFFF",
    ],
    "radius_dic": [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
    "bonds1": [
        [154.0, 147.0, 143.0, 182.0, 0.0, 194.0, 177.0, 184.0, 214.0, 135.0],
        [147.0, 145.0, 140.0, 168.0, 0.0, 214.0, 175.0, 177.0, 222.0, 136.0],
        [143.0, 140.0, 148.0, 151.0, 0.0, 172.0, 164.0, 163.0, 194.0, 142.0],
        [182.0, 168.0, 151.0, 204.0, 0.0, 225.0, 207.0, 210.0, 234.0, 158.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 175.0, 0.0, 0.0, 0.0],
        [194.0, 214.0, 172.0, 225.0, 0.0, 228.0, 214.0, 222.0, 0.0, 178.0],
        [177.0, 175.0, 164.0, 207.0, 175.0, 214.0, 199.0, 203.0, 0.0, 166.0],
        [184.0, 177.0, 163.0, 210.0, 0.0, 222.0, 203.0, 221.0, 0.0, 156.0],
        [214.0, 222.0, 194.0, 234.0, 0.0, 0.0, 0.0, 0.0, 266.0, 187.0],
        [135.0, 136.0, 142.0, 158.0, 0.0, 178.0, 166.0, 156.0, 187.0, 142.0],
    ],
    "bonds2": [
        [134.0, 129.0, 120.0, 160.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [129.0, 125.0, 121.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [120.0, 121.0, 121.0, 0.0, 0.0, 0.0, 0.0, 150.0, 0.0, 0.0],
        [160.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 186.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 150.0, 186.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ],
    "bonds3": [
        [120.0, 116.0, 113.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [116.0, 110.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [113.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ],
    "lennard_jones_rm": [
        [120.0, 116.0, 113.0, 160.0, 133.0, 194.0, 177.0, 184.0, 214.0, 135.0],
        [116.0, 110.0, 121.0, 168.0, 127.0, 214.0, 175.0, 177.0, 222.0, 136.0],
        [113.0, 121.0, 121.0, 151.0, 126.0, 172.0, 164.0, 150.0, 194.0, 142.0],
        [160.0, 168.0, 151.0, 204.0, 167.0, 225.0, 207.0, 186.0, 234.0, 158.0],
        [133.0, 127.0, 126.0, 167.0, 146.0, 182.0, 175.0, 167.0, 198.0, 126.0],
        [194.0, 214.0, 172.0, 225.0, 182.0, 228.0, 214.0, 222.0, 234.0, 178.0],
        [177.0, 175.0, 164.0, 207.0, 175.0, 214.0, 199.0, 203.0, 218.0, 166.0],
        [184.0, 177.0, 150.0, 186.0, 167.0, 222.0, 203.0, 221.0, 219.0, 156.0],
        [214.0, 222.0, 194.0, 234.0, 198.0, 234.0, 218.0, 219.0, 266.0, 187.0],
        [135.0, 136.0, 142.0, 158.0, 126.0, 178.0, 166.0, 156.0, 187.0, 142.0],
    ],
    "atom_hist": {
        "C": 545542,
        "N": 90205,
        "O": 132965,
        "S": 9342,
        "B": 109,
        "Br": 1424,
        "Cl": 5516,
        "P": 5154,
        "I": 445,
        "F": 9742,
    },
    "aa_hist": {
        "A": 109798,
        "C": 31556,
        "D": 83921,
        "E": 79405,
        "F": 97083,
        "G": 139319,
        "H": 62661,
        "I": 99008,
        "K": 62403,
        "L": 155105,
        "M": 59977,
        "N": 70437,
        "P": 58833,
        "Q": 48254,
        "R": 74215,
        "S": 103286,
        "T": 90972,
        "V": 119954,
        "W": 42017,
        "Y": 90596,
    },
}

dataset_params["crossdock_full"] = {
    "atom_encoder": {
        "C": 0,
        "N": 1,
        "O": 2,
        "S": 3,
        "B": 4,
        "Br": 5,
        "Cl": 6,
        "P": 7,
        "I": 8,
        "F": 9,
        "others": 10,
    },
    "atom_decoder": ["C", "N", "O", "S", "B", "Br", "Cl", "P", "I", "F", "others"],
    "aa_encoder": {
        "C": 0,
        "N": 1,
        "O": 2,
        "S": 3,
        "B": 4,
        "Br": 5,
        "Cl": 6,
        "P": 7,
        "I": 8,
        "F": 9,
        "others": 10,
    },
    "aa_decoder": ["C", "N", "O", "S", "B", "Br", "Cl", "P", "I", "F", "others"],
    "colors_dic": [
        "#33ff33",
        "#3333ff",
        "#ff4d4d",
        "#e6c540",
        "#ffb5b5",
        "#A62929",
        "#1FF01F",
        "#ff8000",
        "#940094",
        "#B3FFFF",
        "#ffb5b5",
    ],
    "radius_dic": [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
    "bonds1": [
        [154.0, 147.0, 143.0, 182.0, 0.0, 194.0, 177.0, 184.0, 214.0, 135.0, 0.0],
        [147.0, 145.0, 140.0, 168.0, 0.0, 214.0, 175.0, 177.0, 222.0, 136.0, 0.0],
        [143.0, 140.0, 148.0, 151.0, 0.0, 172.0, 164.0, 163.0, 194.0, 142.0, 0.0],
        [182.0, 168.0, 151.0, 204.0, 0.0, 225.0, 207.0, 210.0, 234.0, 158.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 175.0, 0.0, 0.0, 0.0, 0.0],
        [194.0, 214.0, 172.0, 225.0, 0.0, 228.0, 214.0, 222.0, 0.0, 178.0, 0.0],
        [177.0, 175.0, 164.0, 207.0, 175.0, 214.0, 199.0, 203.0, 0.0, 166.0, 0.0],
        [184.0, 177.0, 163.0, 210.0, 0.0, 222.0, 203.0, 221.0, 0.0, 156.0, 0.0],
        [214.0, 222.0, 194.0, 234.0, 0.0, 0.0, 0.0, 0.0, 266.0, 187.0, 0.0],
        [135.0, 136.0, 142.0, 158.0, 0.0, 178.0, 166.0, 156.0, 187.0, 142.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ],
    "bonds2": [
        [134.0, 129.0, 120.0, 160.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [129.0, 125.0, 121.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [120.0, 121.0, 121.0, 0.0, 0.0, 0.0, 0.0, 150.0, 0.0, 0.0, 0.0],
        [160.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 186.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 150.0, 186.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ],
    "bonds3": [
        [120.0, 116.0, 113.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [116.0, 110.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [113.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ],
    "lennard_jones_rm": [
        [120.0, 116.0, 113.0, 160.0, 133.0, 194.0, 177.0, 184.0, 214.0, 135.0, 0.0],
        [116.0, 110.0, 121.0, 168.0, 127.0, 214.0, 175.0, 177.0, 222.0, 136.0, 0.0],
        [113.0, 121.0, 121.0, 151.0, 126.0, 172.0, 164.0, 150.0, 194.0, 142.0, 0.0],
        [160.0, 168.0, 151.0, 204.0, 167.0, 225.0, 207.0, 186.0, 234.0, 158.0, 0.0],
        [133.0, 127.0, 126.0, 167.0, 146.0, 182.0, 175.0, 167.0, 198.0, 126.0, 0.0],
        [194.0, 214.0, 172.0, 225.0, 182.0, 228.0, 214.0, 222.0, 234.0, 178.0, 0.0],
        [177.0, 175.0, 164.0, 207.0, 175.0, 214.0, 199.0, 203.0, 218.0, 166.0, 0.0],
        [184.0, 177.0, 150.0, 186.0, 167.0, 222.0, 203.0, 221.0, 219.0, 156.0, 0.0],
        [214.0, 222.0, 194.0, 234.0, 198.0, 234.0, 218.0, 219.0, 266.0, 187.0, 0.0],
        [135.0, 136.0, 142.0, 158.0, 126.0, 178.0, 166.0, 156.0, 187.0, 142.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ],
    "atom_hist": {
        "C": 1570767,
        "N": 273858,
        "O": 396837,
        "S": 26352,
        "B": 0,
        "Br": 0,
        "Cl": 15058,
        "P": 25994,
        "I": 0,
        "F": 30687,
        "others": 0,
    },
    "aa_hist": {
        "C": 23302704,
        "N": 6093090,
        "O": 6701210,
        "S": 276805,
        "B": 0,
        "Br": 0,
        "Cl": 0,
        "P": 0,
        "I": 0,
        "F": 0,
        "others": 0,
    },
}

dataset_params["crossdock"] = {
    "atom_encoder": {
        "C": 0,
        "N": 1,
        "O": 2,
        "S": 3,
        "B": 4,
        "Br": 5,
        "Cl": 6,
        "P": 7,
        "I": 8,
        "F": 9,
    },
    "atom_decoder": ["C", "N", "O", "S", "B", "Br", "Cl", "P", "I", "F"],
    "aa_encoder": {
        "A": 0,
        "C": 1,
        "D": 2,
        "E": 3,
        "F": 4,
        "G": 5,
        "H": 6,
        "I": 7,
        "K": 8,
        "L": 9,
        "M": 10,
        "N": 11,
        "P": 12,
        "Q": 13,
        "R": 14,
        "S": 15,
        "T": 16,
        "V": 17,
        "W": 18,
        "Y": 19,
    },
    "aa_decoder": [
        "A",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "I",
        "K",
        "L",
        "M",
        "N",
        "P",
        "Q",
        "R",
        "S",
        "T",
        "V",
        "W",
        "Y",
    ],
    # PyMOL colors, see: https://pymolwiki.org/index.php/Color_Values#Chemical_element_colours
    "colors_dic": [
        "#33ff33",
        "#3333ff",
        "#ff4d4d",
        "#e6c540",
        "#ffb5b5",
        "#A62929",
        "#1FF01F",
        "#ff8000",
        "#940094",
        "#B3FFFF",
    ],
    "radius_dic": [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
    "bonds1": [
        [154.0, 147.0, 143.0, 182.0, 0.0, 194.0, 177.0, 184.0, 214.0, 135.0],
        [147.0, 145.0, 140.0, 168.0, 0.0, 214.0, 175.0, 177.0, 222.0, 136.0],
        [143.0, 140.0, 148.0, 151.0, 0.0, 172.0, 164.0, 163.0, 194.0, 142.0],
        [182.0, 168.0, 151.0, 204.0, 0.0, 225.0, 207.0, 210.0, 234.0, 158.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 175.0, 0.0, 0.0, 0.0],
        [194.0, 214.0, 172.0, 225.0, 0.0, 228.0, 214.0, 222.0, 0.0, 178.0],
        [177.0, 175.0, 164.0, 207.0, 175.0, 214.0, 199.0, 203.0, 0.0, 166.0],
        [184.0, 177.0, 163.0, 210.0, 0.0, 222.0, 203.0, 221.0, 0.0, 156.0],
        [214.0, 222.0, 194.0, 234.0, 0.0, 0.0, 0.0, 0.0, 266.0, 187.0],
        [135.0, 136.0, 142.0, 158.0, 0.0, 178.0, 166.0, 156.0, 187.0, 142.0],
    ],
    "bonds2": [
        [134.0, 129.0, 120.0, 160.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [129.0, 125.0, 121.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [120.0, 121.0, 121.0, 0.0, 0.0, 0.0, 0.0, 150.0, 0.0, 0.0],
        [160.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 186.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 150.0, 186.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ],
    "bonds3": [
        [120.0, 116.0, 113.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [116.0, 110.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [113.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ],
    "lennard_jones_rm": [
        [120.0, 116.0, 113.0, 160.0, 133.0, 194.0, 177.0, 184.0, 214.0, 135.0],
        [116.0, 110.0, 121.0, 168.0, 127.0, 214.0, 175.0, 177.0, 222.0, 136.0],
        [113.0, 121.0, 121.0, 151.0, 126.0, 172.0, 164.0, 150.0, 194.0, 142.0],
        [160.0, 168.0, 151.0, 204.0, 167.0, 225.0, 207.0, 186.0, 234.0, 158.0],
        [133.0, 127.0, 126.0, 167.0, 146.0, 182.0, 175.0, 167.0, 198.0, 126.0],
        [194.0, 214.0, 172.0, 225.0, 182.0, 228.0, 214.0, 222.0, 234.0, 178.0],
        [177.0, 175.0, 164.0, 207.0, 175.0, 214.0, 199.0, 203.0, 218.0, 166.0],
        [184.0, 177.0, 150.0, 186.0, 167.0, 222.0, 203.0, 221.0, 219.0, 156.0],
        [214.0, 222.0, 194.0, 234.0, 198.0, 234.0, 218.0, 219.0, 266.0, 187.0],
        [135.0, 136.0, 142.0, 158.0, 126.0, 178.0, 166.0, 156.0, 187.0, 142.0],
    ],
    "atom_hist": {
        "C": 1570032,
        "N": 273792,
        "O": 396623,
        "S": 26339,
        "B": 0,
        "Br": 0,
        "Cl": 15055,
        "P": 25975,
        "I": 0,
        "F": 30673,
    },
    "aa_hist": {
        "A": 277175,
        "C": 92406,
        "D": 254046,
        "E": 201833,
        "F": 234995,
        "G": 376966,
        "H": 147704,
        "I": 290683,
        "K": 173210,
        "L": 421883,
        "M": 157813,
        "N": 174241,
        "P": 148581,
        "Q": 120232,
        "R": 173848,
        "S": 274430,
        "T": 247605,
        "V": 326134,
        "W": 88552,
        "Y": 226668,
    },
}


dataset_params["pdbbind"] = {
    "atom_encoder": {
        "C": 0,
        "N": 1,
        "O": 2,
        "S": 3,
        "B": 4,
        "Br": 5,
        "Cl": 6,
        "P": 7,
        "I": 8,
        "F": 9,
    },
    "atom_decoder": ["C", "N", "O", "S", "B", "Br", "Cl", "P", "I", "F"],
    "aa_encoder": {
        "A": 0,
        "C": 1,
        "D": 2,
        "E": 3,
        "F": 4,
        "G": 5,
        "H": 6,
        "I": 7,
        "K": 8,
        "L": 9,
        "M": 10,
        "N": 11,
        "P": 12,
        "Q": 13,
        "R": 14,
        "S": 15,
        "T": 16,
        "V": 17,
        "W": 18,
        "Y": 19,
    },
    "aa_decoder": [
        "A",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "I",
        "K",
        "L",
        "M",
        "N",
        "P",
        "Q",
        "R",
        "S",
        "T",
        "V",
        "W",
        "Y",
    ],
    # PyMOL colors, see: https://pymolwiki.org/index.php/Color_Values#Chemical_element_colours
    "colors_dic": [
        "#33ff33",
        "#3333ff",
        "#ff4d4d",
        "#e6c540",
        "#ffb5b5",
        "#A62929",
        "#1FF01F",
        "#ff8000",
        "#940094",
        "#B3FFFF",
    ],
    "radius_dic": [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
    "bonds1": [
        [154.0, 147.0, 143.0, 182.0, 0.0, 194.0, 177.0, 184.0, 214.0, 135.0],
        [147.0, 145.0, 140.0, 168.0, 0.0, 214.0, 175.0, 177.0, 222.0, 136.0],
        [143.0, 140.0, 148.0, 151.0, 0.0, 172.0, 164.0, 163.0, 194.0, 142.0],
        [182.0, 168.0, 151.0, 204.0, 0.0, 225.0, 207.0, 210.0, 234.0, 158.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 175.0, 0.0, 0.0, 0.0],
        [194.0, 214.0, 172.0, 225.0, 0.0, 228.0, 214.0, 222.0, 0.0, 178.0],
        [177.0, 175.0, 164.0, 207.0, 175.0, 214.0, 199.0, 203.0, 0.0, 166.0],
        [184.0, 177.0, 163.0, 210.0, 0.0, 222.0, 203.0, 221.0, 0.0, 156.0],
        [214.0, 222.0, 194.0, 234.0, 0.0, 0.0, 0.0, 0.0, 266.0, 187.0],
        [135.0, 136.0, 142.0, 158.0, 0.0, 178.0, 166.0, 156.0, 187.0, 142.0],
    ],
    "bonds2": [
        [134.0, 129.0, 120.0, 160.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [129.0, 125.0, 121.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [120.0, 121.0, 121.0, 0.0, 0.0, 0.0, 0.0, 150.0, 0.0, 0.0],
        [160.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 186.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 150.0, 186.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ],
    "bonds3": [
        [120.0, 116.0, 113.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [116.0, 110.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [113.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ],
    "lennard_jones_rm": [
        [120.0, 116.0, 113.0, 160.0, 133.0, 194.0, 177.0, 184.0, 214.0, 135.0],
        [116.0, 110.0, 121.0, 168.0, 127.0, 214.0, 175.0, 177.0, 222.0, 136.0],
        [113.0, 121.0, 121.0, 151.0, 126.0, 172.0, 164.0, 150.0, 194.0, 142.0],
        [160.0, 168.0, 151.0, 204.0, 167.0, 225.0, 207.0, 186.0, 234.0, 158.0],
        [133.0, 127.0, 126.0, 167.0, 146.0, 182.0, 175.0, 167.0, 198.0, 126.0],
        [194.0, 214.0, 172.0, 225.0, 182.0, 228.0, 214.0, 222.0, 234.0, 178.0],
        [177.0, 175.0, 164.0, 207.0, 175.0, 214.0, 199.0, 203.0, 218.0, 166.0],
        [184.0, 177.0, 150.0, 186.0, 167.0, 222.0, 203.0, 221.0, 219.0, 156.0],
        [214.0, 222.0, 194.0, 234.0, 198.0, 234.0, 218.0, 219.0, 266.0, 187.0],
        [135.0, 136.0, 142.0, 158.0, 126.0, 178.0, 166.0, 156.0, 187.0, 142.0],
    ],
    "atom_hist": {
        "C": 312417,
        "N": 63669,
        "O": 77484,
        "S": 4873,
        "B": 183,
        "Br": 450,
        "Cl": 2162,
        "P": 1886,
        "I": 132,
        "F": 3661,
    },
    "aa_hist": {
        "A": 44196,
        "C": 13612,
        "D": 37912,
        "E": 32114,
        "F": 37951,
        "G": 59510,
        "H": 24023,
        "I": 44121,
        "K": 31776,
        "L": 67994,
        "M": 21040,
        "N": 28809,
        "P": 23556,
        "Q": 21029,
        "R": 29685,
        "S": 39793,
        "T": 35862,
        "V": 52548,
        "W": 15784,
        "Y": 37883,
    },
}


from typing import Union, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem
import networkx as nx
from networkx.algorithms import isomorphism
from Bio.PDB.Polypeptide import is_aa


class Queue:
    def __init__(self, max_len=50):
        self.items = []
        self.max_len = max_len

    def __len__(self):
        return len(self.items)

    def add(self, item):
        self.items.insert(0, item)
        if len(self) > self.max_len:
            self.items.pop()

    def mean(self):
        return np.mean(self.items)

    def std(self):
        return np.std(self.items)


def reverse_tensor(x):
    return x[torch.arange(x.size(0) - 1, -1, -1)]


#####


def get_grad_norm(
    parameters: Union[torch.Tensor, Iterable[torch.Tensor]], norm_type: float = 2.0
) -> torch.Tensor:
    """
    Adapted from: https://pytorch.org/docs/stable/_modules/torch/nn/utils/clip_grad.html#clip_grad_norm_
    """

    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]

    norm_type = float(norm_type)

    if len(parameters) == 0:
        return torch.tensor(0.0)

    device = parameters[0].grad.device

    total_norm = torch.norm(
        torch.stack(
            [torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]
        ),
        norm_type,
    )

    return total_norm


def write_xyz_file(coords, atom_types, filename):
    out = f"{len(coords)}\n\n"
    assert len(coords) == len(atom_types)
    for i in range(len(coords)):
        out += f"{atom_types[i]} {coords[i, 0]:.3f} {coords[i, 1]:.3f} {coords[i, 2]:.3f}\n"
    with open(filename, "w") as f:
        f.write(out)


def write_sdf_file(sdf_path, molecules):
    # NOTE Changed to be compatitble with more versions of rdkit
    # with Chem.SDWriter(str(sdf_path)) as w:
    #    for mol in molecules:
    #        w.write(mol)

    w = Chem.SDWriter(str(sdf_path))
    for m in molecules:
        if m is not None:
            w.write(m)

    print(f"Wrote SDF file to {sdf_path}")


def residues_to_atoms(x_ca, dataset_info):
    x = x_ca
    one_hot = F.one_hot(
        torch.tensor(dataset_info["atom_encoder"]["C"], device=x_ca.device),
        num_classes=len(dataset_info["atom_encoder"]),
    ).repeat(*x_ca.shape[:-1], 1)
    return x, one_hot


def get_residue_with_resi(pdb_chain, resi):
    res = [x for x in pdb_chain.get_residues() if x.id[1] == resi]
    assert len(res) == 1
    return res[0]


def get_pocket_from_ligand(pdb_model, ligand_id, dist_cutoff=8.0):
    chain, resi = ligand_id.split(":")
    ligand = get_residue_with_resi(pdb_model[chain], int(resi))
    ligand_coords = torch.from_numpy(
        np.array([a.get_coord() for a in ligand.get_atoms()])
    )

    pocket_residues = []
    for residue in pdb_model.get_residues():
        if residue.id[1] == resi:
            continue  # skip ligand itself

        res_coords = torch.from_numpy(
            np.array([a.get_coord() for a in residue.get_atoms()])
        )
        if (
            is_aa(residue.get_resname(), standard=True)
            and torch.cdist(res_coords, ligand_coords).min() < dist_cutoff
        ):
            pocket_residues.append(residue)

    return pocket_residues


def batch_to_list(data, batch_mask):
    # data_list = []
    # for i in torch.unique(batch_mask):
    #     data_list.append(data[batch_mask == i])
    # return data_list

    # make sure batch_mask is increasing
    idx = torch.argsort(batch_mask)
    batch_mask = batch_mask[idx]
    data = data[idx]

    chunk_sizes = torch.unique(batch_mask, return_counts=True)[1].tolist()
    return torch.split(data, chunk_sizes)


def num_nodes_to_batch_mask(n_samples, num_nodes, device):
    assert isinstance(num_nodes, int) or len(num_nodes) == n_samples

    if isinstance(num_nodes, torch.Tensor):
        num_nodes = num_nodes.to(device)

    sample_inds = torch.arange(n_samples, device=device)

    return torch.repeat_interleave(sample_inds, num_nodes)


def rdmol_to_nxgraph(rdmol):
    graph = nx.Graph()
    for atom in rdmol.GetAtoms():
        # Add the atoms as nodes
        graph.add_node(atom.GetIdx(), atom_type=atom.GetAtomicNum())

    # Add the bonds as edges
    for bond in rdmol.GetBonds():
        graph.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())

    return graph


def calc_rmsd(mol_a, mol_b):
    """Calculate RMSD of two molecules with unknown atom correspondence."""
    graph_a = rdmol_to_nxgraph(mol_a)
    graph_b = rdmol_to_nxgraph(mol_b)

    gm = isomorphism.GraphMatcher(
        graph_a, graph_b, node_match=lambda na, nb: na["atom_type"] == nb["atom_type"]
    )

    isomorphisms = list(gm.isomorphisms_iter())
    if len(isomorphisms) < 1:
        return None

    all_rmsds = []
    for mapping in isomorphisms:
        atom_types_a = [atom.GetAtomicNum() for atom in mol_a.GetAtoms()]
        atom_types_b = [
            mol_b.GetAtomWithIdx(mapping[i]).GetAtomicNum()
            for i in range(mol_b.GetNumAtoms())
        ]
        assert atom_types_a == atom_types_b

        conf_a = mol_a.GetConformer()
        coords_a = np.array(
            [conf_a.GetAtomPosition(i) for i in range(mol_a.GetNumAtoms())]
        )
        conf_b = mol_b.GetConformer()
        coords_b = np.array(
            [conf_b.GetAtomPosition(mapping[i]) for i in range(mol_b.GetNumAtoms())]
        )

        diff = coords_a - coords_b
        rmsd = np.sqrt(np.mean(np.sum(diff * diff, axis=1)))
        all_rmsds.append(rmsd)

    if len(isomorphisms) > 1:
        print("More than one isomorphism found. Returning minimum RMSD.")

    return min(all_rmsds)


dataset_info = dataset_params["bindingmoad"]
amino_acid_dict = dataset_info["aa_encoder"]
atom_dict = dataset_info["atom_encoder"]
atom_decoder = dataset_info["atom_decoder"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--basedir", type=Path)
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--qed-thresh", type=float, default=0.3)
    parser.add_argument("--max-occurences", type=int, default=50)
    parser.add_argument("--num-val", type=int, default=300)
    parser.add_argument("--num-test", type=int, default=300)
    parser.add_argument("--dist-cutoff", type=float, default=8.0)
    parser.add_argument("--ca-only", action="store_true")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--make-split", action="store_true")
    args = parser.parse_args()

    pdbdir = args.basedir / "BindingMOAD_2020/"

    # Make output directory
    if args.outdir is None:
        suffix = "" if "H" in atom_dict else "_noH"
        suffix += "_ca_only" if args.ca_only else "_full"
        processed_dir = Path(args.basedir, f"processed{suffix}")
    else:
        processed_dir = args.outdir

    processed_dir.mkdir(exist_ok=True, parents=True)

    if args.make_split:
        # Process the label file
        csv_path = args.basedir / "every.csv"
        ligand_dict = read_label_file(csv_path)
        ligand_dict = compute_druglikeness(ligand_dict)
        filtered_examples = filter_and_flatten(
            ligand_dict, args.qed_thresh, args.max_occurences, args.random_seed
        )
        print(f"{len(filtered_examples)} examples after filtering")

        # Make data split
        data_split = split_by_ec_number(filtered_examples, args.num_val, args.num_test)

    else:
        # Use precomputed data split
        data_split = {}
        for split in ["test", "val", "train"]:
            with open(f"data/moad_{split}.txt", "r") as f:
                pocket_ids = f.read().split(",")
            # (ec-number, protein, molecule tuple)
            data_split[split] = [
                (None, x.split("_")[0][:4], (x.split("_")[1],)) for x in pocket_ids
            ]

    n_train_before = len(data_split["train"])
    n_val_before = len(data_split["val"])
    n_test_before = len(data_split["test"])

    # Read and process PDB files
    n_samples_after = {}
    for split in data_split.keys():
        lig_coords = []
        lig_one_hot = []
        lig_mask = []
        pocket_c_alpha = []
        # pocket_quaternion = []
        pocket_one_hot = []
        pocket_mask = []
        pdb_and_mol_ids = []
        count = 0

        pdb_sdf_dir = processed_dir / split
        pdb_sdf_dir.mkdir(exist_ok=True)

        n_tot = len(data_split[split])
        pair_dict = ligand_list_to_dict(data_split[split])

        tic = time()
        num_failed = 0
        with tqdm(total=n_tot) as pbar:
            for p in pair_dict:
                pdb_successful = set()

                # try all available .bio files
                for pdbfile in sorted(pdbdir.glob(f"{p.lower()}.bio*")):
                    # Skip if all ligands have been processed already
                    if len(pair_dict[p]) == len(pdb_successful):
                        continue

                    pdb_struct = PDBParser(QUIET=True).get_structure("", pdbfile)
                    struct_copy = pdb_struct.copy()

                    n_bio_successful = 0
                    for m in pair_dict[p]:
                        # Skip already processed ligand
                        if m[0] in pdb_successful:
                            continue

                        ligand_name, ligand_chain, ligand_resi = m[0].split(":")
                        ligand_resi = int(ligand_resi)

                        try:
                            ligand_data, pocket_data = process_ligand_and_pocket(
                                pdb_struct,
                                ligand_name,
                                ligand_chain,
                                ligand_resi,
                                dist_cutoff=args.dist_cutoff,
                                ca_only=args.ca_only,
                            )
                        except (
                            KeyError,
                            AssertionError,
                            FileNotFoundError,
                            IndexError,
                            ValueError,
                        ) as e:
                            # print(type(e).__name__, e)
                            continue

                        pdb_and_mol_ids.append(f"{p}_{m[0]}")
                        lig_coords.append(ligand_data["lig_coords"])
                        lig_one_hot.append(ligand_data["lig_one_hot"])
                        lig_mask.append(count * np.ones(len(ligand_data["lig_coords"])))
                        pocket_c_alpha.append(pocket_data["pocket_ca"])
                        # pocket_quaternion.append(
                        #     pocket_data['pocket_quaternion'])
                        pocket_one_hot.append(pocket_data["pocket_one_hot"])
                        pocket_mask.append(
                            count * np.ones(len(pocket_data["pocket_ca"]))
                        )
                        count += 1

                        pdb_successful.add(m[0])
                        n_bio_successful += 1

                        # Save additional files for affinity analysis
                        if split in {"val", "test"}:
                            # remove ligand from receptor
                            try:
                                struct_copy[0][ligand_chain].detach_child(
                                    (f"H_{ligand_name}", ligand_resi, " ")
                                )
                            except KeyError:
                                warnings.warn(
                                    f"Could not find ligand {(f'H_{ligand_name}', ligand_resi, ' ')} in {pdbfile}"
                                )
                                continue

                            # Create SDF file
                            atom_types = [
                                atom_decoder[np.argmax(i)]
                                for i in ligand_data["lig_one_hot"]
                            ]
                            xyz_file = Path(pdb_sdf_dir, "tmp.xyz")
                            write_xyz_file(
                                ligand_data["lig_coords"], atom_types, xyz_file
                            )

                            obConversion = openbabel.OBConversion()
                            obConversion.SetInAndOutFormats("xyz", "sdf")
                            mol = openbabel.OBMol()
                            obConversion.ReadFile(mol, str(xyz_file))
                            xyz_file.unlink()

                            name = f"{p}_{pdbfile.suffix[1:]}_{m[0]}"
                            sdf_file = Path(pdb_sdf_dir, f"{name}.sdf")
                            obConversion.WriteFile(mol, str(sdf_file))

                            # specify pocket residues
                            with open(Path(pdb_sdf_dir, f"{name}.txt"), "w") as f:
                                f.write(" ".join(pocket_data["pocket_ids"]))

                    if split in {"val", "test"} and n_bio_successful > 0:
                        # create receptor PDB file
                        pdb_file_out = Path(
                            pdb_sdf_dir, f"{p}_{pdbfile.suffix[1:]}.pdb"
                        )
                        io = PDBIO()
                        io.set_structure(struct_copy)
                        io.save(str(pdb_file_out))

                pbar.update(len(pair_dict[p]))
                num_failed += len(pair_dict[p]) - len(pdb_successful)
                pbar.set_description(f"#failed: {num_failed}")

        lig_coords = np.concatenate(lig_coords, axis=0)
        lig_one_hot = np.concatenate(lig_one_hot, axis=0)
        lig_mask = np.concatenate(lig_mask, axis=0)
        pocket_c_alpha = np.concatenate(pocket_c_alpha, axis=0)
        # pocket_quaternion = np.concatenate(pocket_quaternion, axis=0)
        pocket_one_hot = np.concatenate(pocket_one_hot, axis=0)
        pocket_mask = np.concatenate(pocket_mask, axis=0)

        # saveall(processed_dir / f'{split}.npz', pdb_and_mol_ids, lig_coords,
        #         lig_one_hot, lig_mask, pocket_c_alpha, pocket_quaternion,
        #         pocket_one_hot, pocket_mask)
        np.savez(
            processed_dir / f"{split}.npz",
            names=pdb_and_mol_ids,
            lig_coords=lig_coords,
            lig_one_hot=lig_one_hot,
            lig_mask=lig_mask,
            pocket_c_alpha=pocket_c_alpha,
            pocket_one_hot=pocket_one_hot,
            pocket_mask=pocket_mask,
        )

        n_samples_after[split] = len(pdb_and_mol_ids)
        print(f"Processing {split} set took {(time() - tic)/60.0:.2f} minutes")

    # --------------------------------------------------------------------------
    # Compute statistics & additional information
    # --------------------------------------------------------------------------
    with np.load(processed_dir / "train.npz", allow_pickle=True) as data:
        lig_mask = data["lig_mask"]
        pocket_mask = data["pocket_mask"]
        lig_coords = data["lig_coords"]
        lig_one_hot = data["lig_one_hot"]
        pocket_one_hot = data["pocket_one_hot"]

    # Compute SMILES for all training examples
    train_smiles = compute_smiles(lig_coords, lig_one_hot, lig_mask)
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
        lig_one_hot, pocket_one_hot, atom_dict, amino_acid_dict
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
    summary_string += f"'bonds1': {bonds1.tolist()}\n"
    summary_string += f"'bonds2': {bonds2.tolist()}\n"
    summary_string += f"'bonds3': {bonds3.tolist()}\n"
    summary_string += f"'lennard_jones_rm': {rm_LJ.tolist()}\n"
    summary_string += f"'atom_hist': {atom_hist}\n"
    summary_string += f"'aa_hist': {aa_hist}\n"
    summary_string += f"'n_nodes': {n_nodes.tolist()}\n"

    # Write summary to text file
    with open(processed_dir / "summary.txt", "w") as f:
        f.write(summary_string)

    # Print summary
    print(summary_string)
