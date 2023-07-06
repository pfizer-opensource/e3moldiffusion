from collections import Counter
import torch
from rdkit import Chem, RDLogger
from rdkit.Geometry import Point3D
from torchmetrics import MaxMetric, MeanMetric

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


def check_stability(molecule, dataset_info, debug=False, atom_decoder=None, smiles=None):
    """ molecule: Molecule object. """
    device = molecule.atom_types.device
    if atom_decoder is None:
        atom_decoder = dataset_info['atom_decoder']

    atom_types = molecule.atom_types
    edge_types = molecule.bond_types

    edge_types[edge_types == 4] = 1.5
    edge_types[edge_types < 0] = 0

    valencies = torch.sum(edge_types, dim=-1).long()

    n_stable_bonds = 0
    mol_stable = True
    for i, (atom_type, valency, charge) in enumerate(zip(atom_types, valencies, molecule.charges)):
        atom_type = atom_type.item()
        valency = valency.item()
        charge = charge.item()
        possible_bonds = allowed_bonds[atom_decoder[atom_type]]
        if type(possible_bonds) == int:
            is_stable = possible_bonds == valency
        elif type(possible_bonds) == dict:
            expected_bonds = possible_bonds[charge] if charge in possible_bonds.keys() else possible_bonds[0]
            is_stable = expected_bonds == valency if type(expected_bonds) == int else valency in expected_bonds
        else:
            is_stable = valency in possible_bonds
        if not is_stable:
            mol_stable = False
        if not is_stable and debug:
            if smiles is not None:
                print(smiles)
            print(f"Invalid atom {atom_decoder[atom_type]}: valency={valency}, charge={charge}")
            print()
        n_stable_bonds += int(is_stable)

    return torch.tensor([mol_stable], dtype=torch.float, device=device),\
           torch.tensor([n_stable_bonds], dtype=torch.float, device=device),\
           len(atom_types)
           
class BasicMolecularMetrics(object):
    def __init__(self, dataset_info, smiles_train=None, device="cpu"):
        self.atom_decoder = dataset_info["atom_decoder"]
        self.dataset_info = dataset_info

        # Retrieve dataset smiles only for qm9 currently.
        self.dataset_smiles_list = set(smiles_train)

        self.atom_stable = MeanMetric().to(device)
        self.mol_stable = MeanMetric().to(device)

        # Retrieve dataset smiles only for qm9 currently.
        self.train_smiles = set(smiles_train)
        self.validity_metric = MeanMetric().to(device)
        self.uniqueness = MeanMetric().to(device)
        self.novelty = MeanMetric().to(device)
        self.mean_components = MeanMetric().to(device)
        self.max_components = MaxMetric().to(device)

    def reset(self):
        for metric in [
            self.atom_stable,
            self.mol_stable,
            self.validity_metric,
            self.uniqueness,
            self.novelty,
            self.mean_components,
            self.max_components,
        ]:
            metric.reset()

    def compute_validity(self, generated, local_rank=0):
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
        if local_rank == 0:
            print(
                f"Error messages: AtomValence {error_message[1]}, Kekulize {error_message[2]}, other {error_message[3]}, "
                f" -- No error {error_message[-1]}"
            )
        self.validity_metric.update(
            value=len(valid) / len(generated), weight=len(generated)
        )
        num_components = torch.tensor(
            num_components, device=self.mean_components.device
        )
        self.mean_components.update(num_components)
        self.max_components.update(num_components)
        not_connected = 100.0 * error_message[4] / len(generated)
        connected_components = 100.0 - not_connected
        return valid, connected_components, all_smiles, error_message

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

    def evaluate(self, generated, local_rank):
        """generated: list of pairs (positions: n x 3, atom_types: n [int])
        the positions and atom types should already be masked."""
        # Validity
        valid, connected_components, all_smiles, error_message = self.compute_validity(
            generated, local_rank=local_rank
        )

        validity = self.validity_metric.compute()
        uniqueness, novelty = 0, 0
        mean_components = self.mean_components.compute()
        max_components = self.max_components.compute()

        # Uniqueness
        if len(valid) > 0:
            unique = list(set(valid))
            self.uniqueness.update(value=len(unique) / len(valid), weight=len(valid))
            uniqueness = self.uniqueness.compute()

            if self.train_smiles is not None:
                novel = []
                for smiles in unique:
                    if smiles not in self.train_smiles:
                        novel.append(smiles)
                self.novelty.update(value=len(novel) / len(unique), weight=len(unique))
            novelty = self.novelty.compute()

        num_molecules = int(self.validity_metric.weight.item())
        if local_rank == 0:
            print(f"Validity over {num_molecules} molecules:" f" {validity * 100 :.2f}%")
            print(
                f"Number of connected components of {num_molecules} molecules: "
                f"mean:{mean_components:.2f} max:{max_components:.2f}"
            )
            print(
                f"Connected components of {num_molecules} molecules: "
                f"{connected_components:.2f}"
            )

        return all_smiles, validity, novelty, uniqueness

    def __call__(self, molecules: list, local_rank=0):
        # Atom and molecule stability
        if local_rank == 0:
            print(f"Analyzing molecule stability on ...")
        for i, mol in enumerate(molecules):
            mol_stable, at_stable, num_bonds = check_stability(
                mol, self.dataset_info
            )
            self.mol_stable.update(value=mol_stable)
            self.atom_stable.update(value=at_stable / num_bonds, weight=num_bonds)

        stability_dict = {
            "mol_stable": self.mol_stable.compute().item(),
            "atm_stable": self.atom_stable.compute().item(),
        }
        # Validity, uniqueness, novelty
        all_generated_smiles, validity, novelty, uniqueness = self.evaluate(
            molecules, local_rank=local_rank
        )
        # Save in any case in the graphs folder
        validity_dict = {
            "validity": validity.item(),
            "novelty": novelty,
            "uniqueness": uniqueness,
        }
        self.reset()
        return stability_dict, validity_dict, all_generated_smiles


def analyze_stability_for_molecules(
    molecule_list, dataset_info, smiles_train, local_rank, device="cuda"
):
    metrics = BasicMolecularMetrics(dataset_info, smiles_train=smiles_train, device=device)
    stability_dict, validity_dict, sampled_smiles = metrics(molecule_list, local_rank=local_rank)
    # print("Unique molecules:", rdkit_metrics[1])
    return stability_dict, validity_dict, sampled_smiles