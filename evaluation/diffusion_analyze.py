from rdkit import Chem
from evaluation.rdkit_functions import BasicMolecularMetrics, build_molecule
import torch
import matplotlib
import torch.nn.functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sp_stats
from evaluation import bond_analyze
from openbabel import openbabel
import tempfile
from rdkit.Chem.rdForceFieldHelpers import UFFOptimizeMolecule, UFFHasAllMoleculeParams
import warnings

openbabel.obErrorLog.SetOutputLevel(0)

use_rdkit = True


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
    "S": {"P": 186},
}


bonds3 = {
    "C": {"C": 120, "N": 116, "O": 113},
    "N": {"C": 116, "N": 110},
    "O": {"C": 113},
}

margin1, margin2, margin3 = 10, 5, 3

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


def get_bond_order(atom1, atom2, distance, check_exists=False):
    distance = 100 * distance  # We change the metric

    # Check exists for large molecules where some atom pairs do not have a
    # typical bond length.
    if check_exists:
        if atom1 not in bonds1:
            return 0
        if atom2 not in bonds1[atom1]:
            return 0

    # margin1, margin2 and margin3 have been tuned to maximize the stability of
    # the QM9 true samples.
    if distance < bonds1[atom1][atom2] + margin1:
        # Check if atoms in bonds2 dictionary.
        if atom1 in bonds2 and atom2 in bonds2[atom1]:
            thr_bond2 = bonds2[atom1][atom2] + margin2
            if distance < thr_bond2:
                if atom1 in bonds3 and atom2 in bonds3[atom1]:
                    thr_bond3 = bonds3[atom1][atom2] + margin3
                    if distance < thr_bond3:
                        return 3  # Triple
                return 2  # Double
        return 1  # Single
    return 0  # No bond


def single_bond_only(threshold, length, margin1=5):
    if length < threshold + margin1:
        return 1
    return 0


def geom_predictor(p, l, margin1=5, limit_bonds_to_one=False):
    """p: atom pair (couple of str)
    l: bond length (float)"""
    bond_order = get_bond_order(p[0], p[1], l, check_exists=True)

    # If limit_bonds_to_one is enabled, every bond type will return 1.
    if limit_bonds_to_one:
        return 1 if bond_order > 0 else 0
    else:
        return bond_order


class Histogram_discrete:
    def __init__(self, name="histogram"):
        self.name = name
        self.bins = {}

    def add(self, elements):
        for e in elements:
            if e in self.bins:
                self.bins[e] += 1
            else:
                self.bins[e] = 1

    def normalize(self):
        total = 0.0
        for key in self.bins:
            total += self.bins[key]
        for key in self.bins:
            self.bins[key] = self.bins[key] / total

    def plot(self, save_path=None):
        width = 1  # the width of the bars
        fig, ax = plt.subplots()
        x, y = [], []
        for key in self.bins:
            x.append(key)
            y.append(self.bins[key])

        ax.bar(x, y, width)
        plt.title(self.name)
        if save_path is not None:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close()


class Histogram_cont:
    def __init__(
        self, num_bins=100, range=(0.0, 13.0), name="histogram", ignore_zeros=False
    ):
        self.name = name
        self.bins = [0] * num_bins
        self.range = range
        self.ignore_zeros = ignore_zeros

    def add(self, elements):
        for e in elements:
            if not self.ignore_zeros or e > 1e-8:
                i = int(float(e) / self.range[1] * len(self.bins))
                i = min(i, len(self.bins) - 1)
                self.bins[i] += 1

    def plot(self, save_path=None):
        width = (self.range[1] - self.range[0]) / len(
            self.bins
        )  # the width of the bars
        fig, ax = plt.subplots()

        x = (
            np.linspace(self.range[0], self.range[1], num=len(self.bins) + 1)[:-1]
            + width / 2
        )
        ax.bar(x, self.bins, width)
        plt.title(self.name)

        if save_path is not None:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close()

    def plot_both(self, hist_b, save_path=None, wandb=None):
        ## TO DO: Check if the relation of bins and linspace is correct
        hist_a = normalize_histogram(self.bins)
        hist_b = normalize_histogram(hist_b)

        # width = (self.range[1] - self.range[0]) / len(self.bins)  # the width of the bars
        fig, ax = plt.subplots()
        x = np.linspace(self.range[0], self.range[1], num=len(self.bins) + 1)[:-1]
        ax.step(x, hist_b)
        ax.step(x, hist_a)
        ax.legend(["True", "Learned"])
        plt.title(self.name)

        if save_path is not None:
            plt.savefig(save_path)
            if wandb is not None:
                if wandb is not None:
                    # Log image(s)
                    im = plt.imread(save_path)
                    wandb.log({save_path: [wandb.Image(im, caption=save_path)]})
        else:
            plt.show()
        plt.close()


def normalize_histogram(hist):
    hist = np.array(hist)
    prob = hist / np.sum(hist)
    return prob


def coord2distances(x):
    x = x.unsqueeze(2)
    x_t = x.transpose(1, 2)
    dist = (x - x_t) ** 2
    dist = torch.sqrt(torch.sum(dist, 3))
    dist = dist.flatten()
    return dist


############################
# Validity and bond analysis
def check_stability(positions, atom_type, dataset_info, debug=False):
    assert len(positions.shape) == 2
    assert positions.shape[1] == 3
    atom_decoder = dataset_info["atom_decoder"]
    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]

    nr_bonds = np.zeros(len(x), dtype="int")

    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            p1 = np.array([x[i], y[i], z[i]])
            p2 = np.array([x[j], y[j], z[j]])
            dist = np.sqrt(np.sum((p1 - p2) ** 2))
            atom1, atom2 = atom_decoder[atom_type[i]], atom_decoder[atom_type[j]]
            pair = sorted([atom_type[i], atom_type[j]])
            if (
                dataset_info["name"] == "qm9"
                or dataset_info["name"] == "qm9_second_half"
                or dataset_info["name"] == "qm9_first_half"
            ):
                order = bond_analyze.get_bond_order(atom1, atom2, dist)

            elif (
                dataset_info["name"] == "drugs"
                or dataset_info["name"] == "aqm"
                or dataset_info["name"] == "uspto"
            ):
                order = bond_analyze.geom_predictor(
                    (atom_decoder[pair[0]], atom_decoder[pair[1]]), dist
                )
            nr_bonds[i] += order
            nr_bonds[j] += order
    nr_stable_bonds = 0
    for atom_type_i, nr_bonds_i in zip(atom_type, nr_bonds):
        possible_bonds = bond_analyze.allowed_bonds[atom_decoder[atom_type_i]]
        if type(possible_bonds) == int:
            is_stable = possible_bonds == nr_bonds_i
        else:
            is_stable = nr_bonds_i in possible_bonds
        if not is_stable and debug:
            print(
                "Invalid bonds for molecule %s with %d bonds"
                % (atom_decoder[atom_type_i], nr_bonds_i)
            )
        nr_stable_bonds += int(is_stable)

    molecule_stable = nr_stable_bonds == len(x)
    return molecule_stable, nr_stable_bonds, len(x)


def write_xyz_file(coords, atom_types, filename):
    out = f"{len(coords)}\n\n"
    assert len(coords) == len(atom_types)
    for i in range(len(coords)):
        out += f"{atom_types[i]} {coords[i, 0]:.3f} {coords[i, 1]:.3f} {coords[i, 2]:.3f}\n"
    with open(filename, "w") as f:
        f.write(out)


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


def make_mol_openbabel(
    positions, atom_types, dataset_info, return_rdkit=False, sanitize=False, relax=False
):
    atom_types = [dataset_info["atom_decoder"][x] for x in atom_types]

    with tempfile.NamedTemporaryFile() as tmp:
        tmp_file = tmp.name

        # Write xyz file
        write_xyz_file(positions, atom_types, tmp_file)

        obConversion = openbabel.OBConversion()
        obConversion.SetInAndOutFormats("xyz", "sdf")
        ob_mol = openbabel.OBMol()
        obConversion.ReadFile(ob_mol, tmp_file)

        obConversion.WriteFile(ob_mol, tmp_file)

        # Read sdf file with RDKit
        mol = Chem.SDMolSupplier(tmp_file, sanitize=False)[0]
        if sanitize:
            Chem.SanitizeMol(mol)
        if relax:
            uff_relax(mol, relax_iter=5)
            if sanitize:
                # sanitize the updated molecule
                Chem.SanitizeMol(mol)
    z, pos = mol_to_tensor(mol)
    if return_rdkit:
        return z, pos, mol
    else:
        return z, pos


def mol_to_tensor(mol):
    pos = torch.tensor(
        np.array(mol.GetConformer(0).GetPositions()), dtype=torch.float32
    )
    atomic_number = []
    for atom in mol.GetAtoms():
        atomic_number.append(atom.GetAtomicNum())
    z = torch.tensor(atomic_number, dtype=torch.int64)
    return z, pos


def sanitize_molecules_openbabel(one_hot, charges, positions, batch, dataset_info):
    num_atoms_prev = 0
    processed_list = []
    atomsxmol = batch.bincount()
    for num_atoms in atomsxmol:
        z = one_hot[num_atoms_prev : num_atoms_prev + num_atoms].argmax(1)
        pos = positions[num_atoms_prev : num_atoms_prev + num_atoms]

        processed_list.append((pos, z))
        num_atoms_prev += num_atoms

    failed_ids = []
    positions = []
    one_hot = []
    n_molecules = 0
    for i, mol in enumerate(processed_list):
        pos, atom_type = mol
        try:
            z, pos = make_mol_openbabel(pos, atom_type, dataset_info)
            one_hot.append(
                F.one_hot(
                    z,
                    num_classes=max(dataset_info["atomic_numbers"]) + 1,
                ).float()[:, dataset_info["atomic_numbers"]]
            )
            positions.append(pos)
            n_molecules += 1
        except:
            failed_ids.append(i)

    atomsxmol = [a for i, a in enumerate(atomsxmol) if i not in failed_ids]
    batch = torch.cat(
        [torch.tensor([0 + i] * n, dtype=torch.int64) for i, n in enumerate(atomsxmol)]
    ).to(batch.device)
    one_hot = torch.cat(one_hot, dim=0).to(batch.device)
    positions = torch.cat(positions, dim=0).to(batch.device)
    charges = charges[: len(positions)]
    return one_hot, charges, positions, batch, n_molecules, failed_ids


def sanitize_and_filter_molecules(
    one_hot, charges, positions, batch, dataset_info, smiles_train
):
    num_atoms_prev = 0
    processed_list = []
    atomsxmol = batch.bincount()
    for num_atoms in atomsxmol:
        z = one_hot[num_atoms_prev : num_atoms_prev + num_atoms].argmax(1)
        pos = positions[num_atoms_prev : num_atoms_prev + num_atoms]

        processed_list.append((pos, z))
        num_atoms_prev += num_atoms

    success_ids = []
    failed_ids = []
    positions = []
    one_hot = []
    n_molecules = 0

    for i, mol in enumerate(processed_list):
        pos, z = mol
        # atom_types, pos, rdmol = make_mol_openbabel(
        #     pos, z, dataset_info, return_rdkit=True
        # )
        smiles = filter_by_validity(z, pos, dataset_info)

        if smiles is not None:
            if smiles in smiles_train:
                failed_ids.append(i)
                continue
        else:
            failed_ids.append(i)
            continue

        one_hot.append(
            F.one_hot(z, num_classes=len(dataset_info["atomic_numbers"])).float()
        )
        success_ids.append(i)

        positions.append(pos)
        n_molecules += 1

    if len(failed_ids) == len(processed_list):
        print("No novel, valid molecule found...\n")
        return None

    atomsxmol = [a for i, a in enumerate(atomsxmol) if i not in failed_ids]
    batch = torch.cat(
        [torch.tensor([0 + i] * n, dtype=torch.int64) for i, n in enumerate(atomsxmol)]
    ).to(batch.device)
    one_hot = torch.cat(one_hot, dim=0).to(batch.device)
    positions = torch.cat(positions, dim=0).to(batch.device)
    charges = charges[success_ids]
    print(
        f"Sampled {round(n_molecules / len(processed_list), 3) * 100}% valid molecules"
    )
    return one_hot, charges, positions, batch, n_molecules, success_ids


def mol2smiles(mol):
    try:
        Chem.SanitizeMol(mol)
    except ValueError:
        return None
    return Chem.MolToSmiles(mol)


def filter_by_validity(z, pos, dataset_info):
    """generated: list of couples (positions, atom_types)"""
    mol = build_molecule(pos, z, dataset_info)
    smiles = mol2smiles(mol)
    if smiles is not None:
        mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
        if len(mol_frags) > 1:
            print("Found fragments, invalid molecule!")
            return None
        else:
            largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
            smiles = mol2smiles(largest_mol)

    return smiles


def filter_by_validity_given_mol(mol):
    mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    if len(mol_frags) > 1:
        print("Found fragments, invalid molecule!")
        return None
    else:
        largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
        Chem.SanitizeMol(largest_mol)
        smiles = Chem.MolToSmiles(largest_mol)

    return smiles


def sanitize_molecules_openbabel_batch(molecule_list, dataset_info):
    one_hot = molecule_list["one_hot"]
    x = molecule_list["x"]
    batch = molecule_list["batch"]
    n_samples = len(x)

    n_molecules = 0

    processed_list = []

    for i in range(n_samples):
        atomsxmol = batch[i].bincount()
        zs = one_hot[i].argmax(1)
        positions = x[i]

        num_atoms_prev = 0
        for num_atoms in atomsxmol:
            z = zs[num_atoms_prev : num_atoms_prev + num_atoms]
            pos = positions[num_atoms_prev : num_atoms_prev + num_atoms]

            processed_list.append((pos, z))
            num_atoms_prev += num_atoms

    failed_ids = []
    positions = []
    batches = []
    one_hot = []
    n_molecules = 0
    for i, mol in enumerate(processed_list):
        pos, atom_type = mol
        try:
            z, pos = make_mol_openbabel(pos, atom_type, dataset_info)
            one_hot.append(
                F.one_hot(
                    z,
                    num_classes=max(dataset_info["atomic_numbers"]) + 1,
                ).float()[:, dataset_info["atomic_numbers"]]
            )
            positions.append(pos)
            batches.append(batch[i])
            n_molecules += 1
        except:
            failed_ids.append(i)

    one_hot = torch.cat(one_hot, dim=0)
    positions = torch.cat(positions, dim=0)
    batches = torch.cat(batches, dim=0)
    molecules = {"one_hot": [one_hot], "x": [positions], "batch": [batches]}

    print(f"\n{n_samples - n_molecules} molecules invalidated by Open Babel!")
    return molecules


def analyze_stability_for_molecules(molecule_list, dataset_info, smiles_train=None):
    one_hot = molecule_list["one_hot"]
    x = molecule_list["x"]
    batch = molecule_list["batch"]
    n_samples = len(x)

    molecule_stable = 0
    nr_stable_bonds = 0
    n_atoms = 0
    n_molecules = 0

    processed_list = []

    for i in range(n_samples):
        atomsxmol = batch[i].bincount()
        zs = one_hot[i].argmax(1)
        positions = x[i]

        num_atoms_prev = 0
        for num_atoms in atomsxmol:
            z = zs[num_atoms_prev : num_atoms_prev + num_atoms]
            pos = positions[num_atoms_prev : num_atoms_prev + num_atoms]

            processed_list.append((pos, z))
            num_atoms_prev += num_atoms

    for mol in processed_list:
        pos, atom_type = mol
        validity_results = check_stability(pos, atom_type, dataset_info)

        molecule_stable += int(validity_results[0])
        nr_stable_bonds += int(validity_results[1])
        n_atoms += int(validity_results[2])
        n_molecules += 1

    # Validity
    fraction_mol_stable = molecule_stable / float(n_molecules)
    fraction_atm_stable = nr_stable_bonds / float(n_atoms)
    validity_dict = {
        "mol_stable": fraction_mol_stable,
        "atm_stable": fraction_atm_stable,
    }

    if use_rdkit:
        metrics = BasicMolecularMetrics(dataset_info, dataset_smiles_list=smiles_train)
        rdkit_metrics = metrics.evaluate(processed_list)
        # print("Unique molecules:", rdkit_metrics[1])
        return validity_dict, rdkit_metrics
    else:
        return validity_dict, None


def analyze_node_distribution(mol_list, save_path):
    hist_nodes = Histogram_discrete("Histogram # nodes (stable molecules)")
    hist_atom_type = Histogram_discrete("Histogram of atom types")

    for molecule in mol_list:
        positions, atom_type = molecule
        hist_nodes.add([positions.shape[0]])
        hist_atom_type.add(atom_type)
    print("Histogram of #nodes")
    print(hist_nodes.bins)
    print("Histogram of # atom types")
    print(hist_atom_type.bins)
    hist_nodes.normalize()
