
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from rdkit.Geometry import Point3D

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

allowed_bonds = {'H': {0: 1, 1: 0, -1: 0},
                 'C': {0: [3, 4], 1: 3, -1: 3},
                 'N': {0: [2, 3], 1: [2, 3, 4], -1: 2},    # In QM9, N+ seems to be present in the form NH+ and NH2+
                 'O': {0: 2, 1: 3, -1: 1},
                 'F': {0: 1, -1: 0},
                 'B': 3, 'Al': 3, 'Si': 4,
                 'P': {0: [3, 5], 1: 4},
                 'S': {0: [2, 6], 1: [2, 3], 2: 4, 3: 5, -1: 3},
                 'Cl': 1, 'As': 3,
                 'Br': {0: 1, 1: 2}, 'I': 1, 'Hg': [1, 2], 'Bi': [3, 5], 'Se': [2, 4, 6]}

bond_dict = [
    None,
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
ATOM_VALENCY = {6: 4, 7: 3, 8: 2, 9: 1, 15: 3, 16: 2, 17: 1, 35: 1, 53: 1}

    
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
                    largest_mol = max(
                        mol_frags, default=mol, key=lambda m: m.GetNumAtoms()
                    )
                    Chem.SanitizeMol(largest_mol)
                    smiles = Chem.MolToSmiles(largest_mol)
                    valid.append(smiles)
                    all_smiles.append(smiles)
                    error_message[-1] += 1
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
            f" -- No error {error_message[-1]}"
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


def analyze_stability_for_molecules(
    molecule_list, dataset_info, smiles_train=None, bonds_given=False, debug=False
):
    molecule_stable = 0
    nr_stable_bonds = 0
    n_atoms = 0
    n_molecules = 0

    for mol in molecule_list:
        if bonds_given:
            validity_results = check_stability_given_bonds(mol, dataset_info, debug=debug)
        else:
            raise NotImplementedError

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
