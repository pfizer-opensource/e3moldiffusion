import tempfile
import warnings
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from openbabel import openbabel, pybel
from rdkit import Chem, RDLogger
from rdkit.Chem.rdForceFieldHelpers import (UFFHasAllMoleculeParams,
                                            UFFOptimizeMolecule)
from rdkit.Geometry import Point3D

from evaluation import bond_analyze

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

allowed_bonds = {
    "H": {0: 1, 1: 0, -1: 0},
    "C": {0: [3, 4], 1: 3, -1: 3},
    "N": {
        0: [2, 3],
        1: [2, 3, 4],
        -1: 2,
    },  # In QM9, N+ seems to be present in the form NH+ and NH2+
    "O": {0: 2, 1: 3, -1: 1},
    "F": {0: 1, -1: 0},
    "B": 3,
    "Al": 3,
    "Si": 4,
    "P": {0: [3, 5], 1: 4},
    "S": {0: [2, 6], 1: [2, 3], 2: 4, 3: 5, -1: 3},
    "Cl": 1,
    "As": 3,
    "Br": {0: 1, 1: 2},
    "I": 1,
    "Hg": [1, 2],
    "Bi": [3, 5],
    "Se": [2, 4, 6],
}
bond_dict = [
    None,
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
ATOM_VALENCY = {6: 4, 7: 3, 8: 2, 9: 1, 15: 3, 16: 2, 17: 1, 35: 1, 53: 1}


class Molecule:
    def __init__(
        self, atom_types, positions, charges=None, bond_types=None, dataset_info=None
    ):
        """
        atom_types: n      LongTensor
        charges: n         LongTensor
        bond_types: n x n  LongTensor
        positions: n x 3   FloatTensor
        atom_decoder: extracted from dataset_infos."""
        atom_decoder = dataset_info["atom_decoder"]
        assert atom_types.dim() == 1 and atom_types.dtype == torch.long, (
            f"shape of atoms {atom_types.shape} " f"and dtype {atom_types.dtype}"
        )
        if bond_types is not None:
            assert bond_types.dim() == 2 and bond_types.dtype == torch.long, (
                f"shape of bonds {bond_types.shape} --" f" {bond_types.dtype}"
            )
            assert len(bond_types.shape) == 2
        assert len(atom_types.shape) == 1
        assert len(positions.shape) == 2
        if charges.ndim == 2:
            charges = charges.squeeze()
        elif charges.shape[0] == 0:
            charges = torch.zeros_like(atom_types)

        self.atom_types = atom_types.long().cpu()
        self.bond_types = bond_types.long().cpu() if bond_types is not None else None
        self.positions = positions.cpu()
        self.charges = charges.cpu()
        self.rdkit_mol = (
             self.build_molecule_given_bonds(atom_decoder)
             if bond_types is not None
             else self.build_molecule(
                 self.positions, self.atom_types, dataset_info
             )  # alternatively xyz_to_mol
         )
        #self.rdkit_mol = self.build_molecule(
        #    self.positions, self.atom_types, dataset_info
        #)
        self.num_nodes = len(atom_types)
        self.num_atom_types = len(atom_decoder)

    def build_molecule_given_bonds(self, atom_decoder, verbose=False):
        if verbose:
            print("building new molecule")

        mol = Chem.RWMol()
        for atom, charge in zip(self.atom_types, self.charges):
            if atom == -1:
                continue
            a = Chem.Atom(atom_decoder[int(atom.item())])
            if charge.item() != 0:
                a.SetFormalCharge(charge.item())
            mol.AddAtom(a)
            if verbose:
                print("Atom added: ", atom.item(), atom_decoder[atom.item()])

        edge_types = torch.triu(self.bond_types, diagonal=1)
        edge_types[edge_types == -1] = 0
        all_bonds = torch.nonzero(edge_types)

        for i, bond in enumerate(all_bonds):
            if bond[0].item() != bond[1].item():
                mol.AddBond(
                    bond[0].item(),
                    bond[1].item(),
                    bond_dict[edge_types[bond[0], bond[1]].item()],
                )
                if verbose:
                    print(
                        "bond added:",
                        bond[0].item(),
                        bond[1].item(),
                        edge_types[bond[0], bond[1]].item(),
                        bond_dict[edge_types[bond[0], bond[1]].item()],
                    )
        try:
            mol = mol.GetMol()
        except Chem.KekulizeException:
            print("Can't kekulize molecule")
            return None

        # Set coordinates
        positions = self.positions.double()
        conf = Chem.Conformer(mol.GetNumAtoms())
        for i in range(mol.GetNumAtoms()):
            conf.SetAtomPosition(
                i,
                Point3D(
                    positions[i][0].item(),
                    positions[i][1].item(),
                    positions[i][2].item(),
                ),
            )
        mol.AddConformer(conf)

        return mol

    def build_molecule(self, positions, atom_types, dataset_info):
        atom_decoder = dataset_info["atom_decoder"]
        X, A, E = self.build_xae_molecule(positions, atom_types, dataset_info)
        mol = Chem.RWMol()
        for atom in X:
            a = Chem.Atom(atom_decoder[atom.item()])
            mol.AddAtom(a)

        all_bonds = torch.nonzero(A)
        for bond in all_bonds:
            mol.AddBond(
                bond[0].item(), bond[1].item(), bond_dict[E[bond[0], bond[1]].item()]
            )
            
        # Set coordinates
        positions = self.positions.double()
        conf = Chem.Conformer(mol.GetNumAtoms())
        for i in range(mol.GetNumAtoms()):
            conf.SetAtomPosition(
                i,
                Point3D(
                    positions[i][0].item(),
                    positions[i][1].item(),
                    positions[i][2].item(),
                ),
            )
        mol.AddConformer(conf)
        
        return mol

    def build_xae_molecule(self, positions, atom_types, dataset_info):
        """Returns a triplet (X, A, E): atom_types, adjacency matrix, edge_types
        args:
        positions: N x 3  (already masked to keep final number nodes)
        atom_types: N
        returns:
        X: N         (int)
        A: N x N     (bool)                  (binary adjacency matrix)
        E: N x N     (int)  (bond type, 0 if no bond) such that A = E.bool()
        """
        atom_decoder = dataset_info["atom_decoder"]
        n = positions.shape[0]
        X = atom_types
        A = torch.zeros((n, n), dtype=torch.bool)
        E = torch.zeros((n, n), dtype=torch.int)

        pos = positions.unsqueeze(0)
        dists = torch.cdist(pos, pos, p=2).squeeze(0)
        for i in range(n):
            for j in range(i):
                pair = sorted([atom_types[i], atom_types[j]])
                if (
                    dataset_info["name"] == "qm9"
                    or dataset_info["name"] == "qm9_second_half"
                    or dataset_info["name"] == "qm9_first_half"
                ):
                    order = bond_analyze.get_bond_order(
                        atom_decoder[pair[0]], atom_decoder[pair[1]], dists[i, j]
                    )
                elif dataset_info["name"] == "geom" or dataset_info["name"] == "aqm":
                    order = bond_analyze.geom_predictor(
                        (atom_decoder[pair[0]], atom_decoder[pair[1]]),
                        dists[i, j],
                        limit_bonds_to_one=True,
                    )
                # TODO: a batched version of get_bond_order to avoid the for loop
                if order > 0:
                    # Warning: the graph should be DIRECTED
                    A[i, j] = 1
                    E[i, j] = order
        return X, A, E


class BasicMolecularMetrics(object):
    def __init__(self, dataset_info, smiles_train=None):
        self.atom_decoder = dataset_info["atom_decoder"]
        self.dataset_info = dataset_info

        # Retrieve dataset smiles only for qm9 currently.
        self.dataset_smiles_list = set(smiles_train)

    def compute_validity(self, generated):
        """generated: list of couples (positions, atom_types)"""
        valid = []
        num_components = []
        all_smiles = []
        error_message = Counter()
        for mol in generated:
            rdmol = mol.rdkit_mol
            if rdmol is not None:
                try:
                    mol_frags = Chem.rdmolops.GetMolFrags(
                        rdmol, asMols=True, sanitizeFrags=False
                    )
                    num_components.append(len(mol_frags))
                    if len(mol_frags) > 1:
                        error_message[4] += 1
                        try:
                            largest_mol = max(
                                mol_frags, default=mol, key=lambda m: m.GetNumAtoms()
                            )
                            Chem.SanitizeMol(largest_mol)
                            smiles = Chem.MolToSmiles(largest_mol)
                            valid.append(smiles)
                            all_smiles.append(smiles)
                        except:
                            continue
                    else:
                        Chem.SanitizeMol(rdmol)
                        smiles = Chem.MolToSmiles(rdmol)
                        valid.append(smiles)
                        all_smiles.append(smiles)
                except Chem.rdchem.AtomValenceException:
                    error_message[1] += 1
                    # print("Valence error in GetmolFrags")
                except Chem.rdchem.KekulizeException:
                    error_message[2] += 1
                    # print("Can't kekulize molecule")
                except Chem.rdchem.AtomKekulizeException or ValueError:
                    error_message[3] += 1
            # else:
            #     error_message[3] += 1
            #     all_smiles.append(None)

        print(
            f"Error messages: AtomValence {error_message[1]}, Kekulize {error_message[2]}, other {error_message[3]}, "
            f"connected_components {error_message[4]}"
            f" -- No error {len(generated) - sum(error_message.values())} / {len(generated)}"
        )
        return (
            valid,
            len(valid) / len(generated),
            np.array(num_components),
            all_smiles,
            error_message,
        )

    def compute_uniqueness(self, valid):
        """valid: list of SMILES strings."""
        return list(set(valid)), len(set(valid)) / len(valid)

    def compute_novelty(self, unique):
        num_novel = 0
        novel = []
        if self.dataset_smiles_list is None:
            print("Dataset smiles is None, novelty computation skipped")
            return 1, 1
        for smiles in unique:
            if smiles not in self.dataset_smiles_list:
                novel.append(smiles)
                num_novel += 1
        return novel, num_novel / len(unique)

    def evaluate(self, generated):
        """generated: list of pairs (positions: n x 3, atom_types: n [int])"""
        (
            valid,
            validity,
            num_components,
            all_smiles,
            error_message,
        ) = self.compute_validity(generated)
        nc_mu = float(num_components.mean()) if len(num_components) > 0 else 0.0
        nc_min = float(num_components.min()) if len(num_components) > 0 else 0.0
        nc_max = float(num_components.max()) if len(num_components) > 0 else 0.0
        print(f"Validity over {len(generated)} molecules: {validity * 100 :.2f}%")
        print(
            f"Number of connected components of {len(generated)} molecules: min:{nc_min:.2f} mean:{nc_mu:.2f} max:{nc_max:.2f}"
        )

        if validity > 0:
            unique, uniqueness = self.compute_uniqueness(valid)
            print(
                f"Uniqueness over {len(valid)} valid molecules: {uniqueness * 100 :.2f}%"
            )

            if self.dataset_smiles_list is not None:
                _, novelty = self.compute_novelty(unique)
                print(
                    f"Novelty over {len(unique)} unique valid molecules: {novelty * 100 :.2f}%"
                )
            else:
                novelty = -1.0
        else:
            novelty = -1.0
            uniqueness = 0.0
            unique = []
        return (
            [validity, uniqueness, novelty],
            unique,
            dict(nc_min=nc_min, nc_max=nc_max, nc_mu=nc_mu),
            all_smiles,
        )


def check_stability_given_bonds(
    molecule, dataset_info, debug=False, atom_decoder=None, smiles=None
):
    """molecule: Molecule object."""
    if atom_decoder is None:
        atom_decoder = dataset_info["atom_decoder"]

    atom_types = molecule.atom_types
    edge_types = molecule.bond_types

    edge_types[edge_types == 4] = 1.5
    edge_types[edge_types < 0] = 0

    valencies = torch.sum(edge_types, dim=-1).long()

    n_stable_bonds = 0
    mol_stable = True
    for i, (atom_type, valency, charge) in enumerate(
        zip(atom_types, valencies, molecule.charges)
    ):
        atom_type = atom_type.item()
        valency = valency.item()
        charge = charge.item()
        possible_bonds = allowed_bonds[atom_decoder[atom_type]]
        if type(possible_bonds) == int:
            is_stable = possible_bonds == valency
        elif type(possible_bonds) == dict:
            expected_bonds = (
                possible_bonds[charge]
                if charge in possible_bonds.keys()
                else possible_bonds[0]
            )
            is_stable = (
                expected_bonds == valency
                if type(expected_bonds) == int
                else valency in expected_bonds
            )
        else:
            is_stable = valency in possible_bonds
        if not is_stable:
            mol_stable = False
        if not is_stable and debug:
            if smiles is not None:
                print(smiles)
            print(
                f"Invalid atom {atom_decoder[atom_type]}: valency={valency}, charge={charge}"
            )
            print()
        n_stable_bonds += int(is_stable)

    return mol_stable, n_stable_bonds, len(atom_types)


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


def xyz_to_mol(positions, atom_types, atom_decoder):
    atom_types = [atom_decoder[x] for x in atom_types]

    with tempfile.NamedTemporaryFile() as tmp:
        tmp_file = tmp.name

        # Write xyz file
        write_xyz_file(positions, atom_types, tmp_file)

        mol = next(pybel.readfile("xyz", tmp_file))
        mol = Chem.MolFromPDBBlock(
            molBlock=mol.write(format="pdb"),
            sanitize=False,
            removeHs=False,
            proximityBonding=True,
        )
    return mol


def make_mol_openbabel(
    positions,
    atom_types,
    atom_decoder,
    sanitize=False,
    relax=False,
    return_mol=False,
):
    atom_types = [atom_decoder[x] for x in atom_types]

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
    if return_mol:
        return mol
    else:
        return mol_to_tensor(mol)


def mol_to_tensor(mol):
    pos = torch.tensor(
        np.array(mol.GetConformer(0).GetPositions()), dtype=torch.float32
    )
    atomic_symbol = []
    for atom in mol.GetAtoms():
        atomic_symbol.append(atom.GetSymbol())
    return atomic_symbol, pos


def sanitize_molecules_openbabel(molecule_list, dataset_info):
    device = molecule_list[0].atom_types.device
    atom_encoder = dataset_info["atom_encoder"]
    atom_decoder = dataset_info["atom_decoder"]

    n_molecules = 0
    failed_ids = []
    new_molecule_list = []
    for i, mol in enumerate(molecule_list):
        pos, atom_type = mol.positions, mol.atom_types
        try:
            z, pos = make_mol_openbabel(pos, atom_type, atom_decoder)
            pos = pos.to(device)
            z = torch.tensor(
                [atom_encoder[s] for s in z],
                dtype=torch.int64,
                device=device,
            )
            molecule = Molecule(
                atom_types=z,
                positions=pos,
                charges=mol.charges,
                bonds=mol.bonds,
                atom_decoder=dataset_info["atom_decoder"],
            )
            new_molecule_list.append(molecule)

            n_molecules += 1
        except:
            failed_ids.append(i)

    print(f"\n{len(molecule_list) - n_molecules} molecules invalidated by Open Babel!")
    return new_molecule_list


def sanitize_molecules_openbabel_batch(molecule_list, dataset_info):
    one_hot = molecule_list["one_hot"]
    x = molecule_list["x"]
    batch = molecule_list["batch"]
    n_samples = len(x)
    n_molecules = 0

    failed_ids = []
    positions = []
    batches = []
    one_hot = []
    n_molecules = 0
    for i, mol in enumerate(molecule_list):
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
    return molecules


def analyze_stability_for_molecules(
    molecule_list, dataset_info, smiles_train=None, bonds_given=False
):
    molecule_stable = 0
    nr_stable_bonds = 0
    n_atoms = 0
    n_molecules = 0

    for mol in molecule_list:
        pos, atom_type = mol.positions, mol.atom_types
        if bonds_given:
            validity_results = check_stability_given_bonds(mol, dataset_info)
        else:
            validity_results = check_stability(pos.cpu(), atom_type.cpu(), dataset_info)

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

    metrics = BasicMolecularMetrics(dataset_info, smiles_train=smiles_train)
    rdkit_metrics = metrics.evaluate(molecule_list)
    # print("Unique molecules:", rdkit_metrics[1])
    return validity_dict, rdkit_metrics
