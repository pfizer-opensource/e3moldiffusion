from collections import Counter
import torch
from rdkit import Chem, RDLogger
from rdkit.Geometry import Point3D
from torchmetrics import MaxMetric, MeanMetric
from experiments.sampling.utils import *
import numpy as np
import itertools
import torch
import pickle
from tqdm import tqdm
from multiprocessing import Pool
from rdkit import Chem, DataStructs
from rdkit.DataStructs import TanimotoSimilarity, BulkTanimotoSimilarity
from rdkit.Chem import AllChem
from rdkit.Chem.QED import qed

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


class BasicMolecularMetrics(object):
    def __init__(self, dataset_info, smiles_train=None, test=False, device="cpu"):
        self.atom_decoder = (
            dataset_info["atom_decoder"]
            if isinstance(dataset_info, dict)
            else dataset_info.atom_decoder
        )
        self.dataset_info = dataset_info

        self.number_samples = 0  # update based on unique generated smiles
        self.train_smiles = smiles_train

        self.train_fps = [
            AllChem.GetMorganFingerprint(Chem.MolFromSmiles(smiles), 2)
            for smiles in smiles_train
        ]
        self.test = test

        self.dataset_smiles_list = set(smiles_train)

        self.atom_stable = MeanMetric().to(device)
        self.mol_stable = MeanMetric().to(device)

        # Retrieve dataset smiles.
        self.validity_metric = MeanMetric().to(device)
        self.uniqueness = MeanMetric().to(device)
        self.novelty = MeanMetric().to(device)
        self.mean_components = MeanMetric().to(device)
        self.max_components = MaxMetric().to(device)
        self.num_nodes_w1 = MeanMetric().to(device)
        self.atom_types_tv = MeanMetric().to(device)
        self.edge_types_tv = MeanMetric().to(device)
        self.charge_w1 = MeanMetric().to(device)
        self.valency_w1 = MeanMetric().to(device)
        self.bond_lengths_w1 = MeanMetric().to(device)
        self.angles_w1 = MeanMetric().to(device)

        self.pc_descriptor_subset = [
            "BertzCT",
            "MolLogP",
            "MolWt",
            "TPSA",
            "NumHAcceptors",
            "NumHDonors",
            "NumRotatableBonds",
            "NumAliphaticRings",
            "NumAromaticRings",
        ]

    def reset(self):
        for metric in [
            self.atom_stable,
            self.mol_stable,
            self.validity_metric,
            self.uniqueness,
            self.novelty,
            self.mean_components,
            self.max_components,
            self.num_nodes_w1,
            self.atom_types_tv,
            self.edge_types_tv,
            self.charge_w1,
            self.valency_w1,
            self.bond_lengths_w1,
            self.angles_w1,
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
            print(
                f"Validity over {num_molecules} molecules:" f" {validity * 100 :.2f}%"
            )
            print(
                f"Number of connected components of {num_molecules} molecules: "
                f"mean:{mean_components:.2f} max:{max_components:.2f}"
            )
            print(
                f"Connected components of {num_molecules} molecules: "
                f"{connected_components:.2f}"
            )

        return all_smiles, validity, novelty, uniqueness, connected_components

    def __call__(self, molecules: list, local_rank=0, return_smiles=False):
        # Atom and molecule stability
        if local_rank == 0:
            print(f"Analyzing molecule stability on ...")
        for i, mol in enumerate(molecules):
            mol_stable, at_stable, num_bonds = check_stability(mol, self.dataset_info)
            self.mol_stable.update(value=mol_stable)
            self.atom_stable.update(value=at_stable / num_bonds, weight=num_bonds)

        stability_dict = {
            "mol_stable": self.mol_stable.compute().item(),
            "atm_stable": self.atom_stable.compute().item(),
        }
        # Validity, uniqueness, novelty
        (
            all_generated_smiles,
            validity,
            novelty,
            uniqueness,
            connected_components,
        ) = self.evaluate(molecules, local_rank=local_rank)
        # Save in any case in the graphs folder

        novelty = novelty if isinstance(novelty, int) else novelty.item()
        uniqueness = uniqueness if isinstance(uniqueness, int) else uniqueness.item()

        validity_dict = {
            "validity": validity.item(),
            "novelty": novelty,
            "uniqueness": uniqueness,
        }

        statistics_dict = self.compute_statistics(molecules, local_rank)
        statistics_dict["connected_components"] = connected_components

        self.number_samples = len(all_generated_smiles)
        self.train_subset = canonicalize_list(
            get_random_subset(self.train_smiles, self.number_samples, seed=42),
            include_stereocenters=False,
        )
        similarity = self.get_bulk_similarity_with_train(molecules)
        diversity = self.get_bulk_diversity(molecules)
        kl_score = self.kl_divergence(all_generated_smiles)
        statistics_dict["similarity"] = similarity
        statistics_dict["diversity"] = diversity
        statistics_dict["kl_score"] = kl_score

        self.reset()

        if not return_smiles:
            all_generated_smiles = None
        return (
            stability_dict,
            validity_dict,
            statistics_dict,
            all_generated_smiles,
        )

    def compute_statistics(self, molecules, local_rank):
        # Compute statistics
        stat = (
            self.dataset_info.statistics["test"]
            if self.test
            else self.dataset_info.statistics["val"]
        )

        self.num_nodes_w1(number_nodes_distance(molecules, stat.num_nodes))

        atom_types_tv, atom_tv_per_class = atom_types_distance(
            molecules, stat.atom_types, save_histogram=self.test
        )
        self.atom_types_tv(atom_types_tv)
        edge_types_tv, bond_tv_per_class, sparsity_level = bond_types_distance(
            molecules, stat.bond_types, save_histogram=self.test
        )
        print(
            f"Sparsity level on local rank {local_rank}: {int(100 * sparsity_level)} %"
        )
        self.edge_types_tv(edge_types_tv)
        charge_w1, charge_w1_per_class = charge_distance(
            molecules, stat.charge_types, stat.atom_types, self.dataset_info
        )
        self.charge_w1(charge_w1)
        valency_w1, valency_w1_per_class = valency_distance(
            molecules, stat.valencies, stat.atom_types, self.dataset_info.atom_encoder
        )
        self.valency_w1(valency_w1)
        bond_lengths_w1, bond_lengths_w1_per_type = bond_length_distance(
            molecules, stat.bond_lengths, stat.bond_types
        )
        self.bond_lengths_w1(bond_lengths_w1)
        if sparsity_level < 0.7:
            if local_rank == 0:
                print(f"Too many edges, skipping angle distance computation.")
            angles_w1 = 0
            angles_w1_per_type = [-1] * len(self.dataset_info.atom_decoder)
        else:
            angles_w1, angles_w1_per_type = angle_distance(
                molecules,
                stat.bond_angles,
                stat.atom_types,
                stat.valencies,
                atom_decoder=self.dataset_info.atom_decoder,
                save_histogram=self.test,
            )
        self.angles_w1(angles_w1)
        statistics_log = {
            "sampling/NumNodesW1": self.num_nodes_w1.compute().item(),
            "sampling/AtomTypesTV": self.atom_types_tv.compute().item(),
            "sampling/EdgeTypesTV": self.edge_types_tv.compute().item(),
            "sampling/ChargeW1": self.charge_w1.compute().item(),
            "sampling/ValencyW1": self.valency_w1.compute().item(),
            "sampling/BondLengthsW1": self.bond_lengths_w1.compute().item(),
            "sampling/AnglesW1": self.angles_w1.compute().item(),
        }
        # if local_rank == 0:
        #     print(
        #         f"Sampling metrics",
        #         {key: round(val.item(), 3) for key, val in statistics_log},
        #     )

        sampling_per_class = False
        if sampling_per_class:
            for i, atom_type in enumerate(self.dataset_info.atom_decoder):
                statistics_log[
                    f"sampling_per_class/{atom_type}_TV"
                ] = atom_tv_per_class[i].item()
                statistics_log[
                    f"sampling_per_class/{atom_type}_ValencyW1"
                ] = valency_w1_per_class[i].item()
                statistics_log[f"sampling_per_class/{atom_type}_BondAnglesW1"] = (
                    angles_w1_per_type[i].item() if angles_w1_per_type[i] != -1 else -1
                )
                statistics_log[
                    f"sampling_per_class/{atom_type}_ChargesW1"
                ] = charge_w1_per_class[i].item()

            for j, bond_type in enumerate(
                ["No bond", "Single", "Double", "Triple", "Aromatic"]
            ):
                statistics_log[
                    f"sampling_per_class/{bond_type}_TV"
                ] = bond_tv_per_class[j].item()
                if j > 0:
                    statistics_log[
                        f"sampling_per_class/{bond_type}_BondLengthsW1"
                    ] = bond_lengths_w1_per_type[j - 1].item()

        return statistics_log

    def get_similarity_with_train(self, molecules, parallel=False):
        mol_fps = [AllChem.GetMorganFingerprint(mol.rdkit_mol, 2) for mol in molecules]
        fp_pair = list(itertools.product(mol_fps, self.train_fps))
        if not parallel:
            similarity_list = []
            for fg1, fg2 in tqdm(fp_pair, desc="Calculate similarity with train"):
                similarity_list.append(get_similarity((fg1, fg2)))
        else:
            with Pool(102) as pool:
                similarity_list = list(
                    tqdm(
                        pool.imap(get_similarity, fp_pair),
                        total=len(mol_fps) * len(self.train_fps),
                    )
                )
        # calculate the max similarity of each mol with train data
        similarity_max = np.reshape(similarity_list, (len(molecules), -1)).max(axis=1)
        return np.mean(similarity_max)

    def get_diversity(self, molecules, parallel=False):
        fps = [AllChem.GetMorganFingerprint(mol.rdkit_mol, 2) for mol in molecules]
        all_fp_pairs = list(itertools.combinations(fps, 2))
        if not parallel:
            similarity_list = []
            for fg1, fg2 in tqdm(all_fp_pairs, desc="Calculate diversity"):
                similarity_list.append(TanimotoSimilarity(fg1, fg2))
        else:
            with Pool(102) as pool:
                similarity_list = pool.imap_unordered(TanimotoSimilarity, all_fp_pairs)
        return 1 - np.mean(similarity_list)

    def get_bulk_similarity_with_train(self, molecules):
        fps = [AllChem.GetMorganFingerprint(mol.rdkit_mol, 2) for mol in molecules]
        scores = []
        for fp in fps:
            scores.append(BulkTanimotoSimilarity(fp, self.train_fps))
        return np.mean(scores)

    def get_bulk_diversity(self, molecules):
        fps = [AllChem.GetMorganFingerprint(mol.rdkit_mol, 2) for mol in molecules]
        scores = []
        for i, fp in enumerate(fps):
            fps_tmp = fps.copy()
            del fps_tmp[i]
            scores.append(BulkTanimotoSimilarity(fp, fps_tmp))
        return 1 - np.mean(scores)

    def kl_divergence(self, generated_smiles):
        # canonicalize_list in order to remove stereo information (also removes duplicates and invalid molecules, but there shouldn't be any)
        unique_molecules = set(
            canonicalize_list(generated_smiles, include_stereocenters=False)
        )

        # first we calculate the descriptors, which are np.arrays of size n_samples x n_descriptors
        d_sampled = calculate_pc_descriptors(
            unique_molecules, self.pc_descriptor_subset
        )
        d_chembl = calculate_pc_descriptors(
            self.train_subset, self.pc_descriptor_subset
        )

        kldivs = {}

        # now we calculate the kl divergence for the float valued descriptors ...
        for i in range(4):
            kldiv = continuous_kldiv(
                X_baseline=d_chembl[:, i], X_sampled=d_sampled[:, i]
            )
            kldivs[self.pc_descriptor_subset[i]] = kldiv

        # ... and for the int valued ones.
        for i in range(4, 9):
            kldiv = discrete_kldiv(X_baseline=d_chembl[:, i], X_sampled=d_sampled[:, i])
            kldivs[self.pc_descriptor_subset[i]] = kldiv

        # pairwise similarity

        chembl_sim = calculate_internal_pairwise_similarities(self.train_subset)
        chembl_sim = chembl_sim.max(axis=1)

        sampled_sim = calculate_internal_pairwise_similarities(unique_molecules)
        sampled_sim = sampled_sim.max(axis=1)

        kldiv_int_int = continuous_kldiv(X_baseline=chembl_sim, X_sampled=sampled_sim)
        kldivs["internal_similarity"] = kldiv_int_int

        # Each KL divergence value is transformed to be in [0, 1].
        # Then their average delivers the final score.
        partial_scores = [np.exp(-score) for score in kldivs.values()]
        score = sum(partial_scores) / len(partial_scores)

        return score


def analyze_stability_for_molecules(
    molecule_list,
    dataset_info,
    smiles_train,
    local_rank,
    return_smiles=False,
    device="cuda",
):
    metrics = BasicMolecularMetrics(
        dataset_info, smiles_train=smiles_train, device=device
    )
    (
        stability_dict,
        validity_dict,
        statistics_dict,
        sampled_smiles,
    ) = metrics(molecule_list, local_rank=local_rank, return_smiles=return_smiles)
    return (
        stability_dict,
        validity_dict,
        statistics_dict,
        sampled_smiles,
    )
