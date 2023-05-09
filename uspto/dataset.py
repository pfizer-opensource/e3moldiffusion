import numpy as np
import os
import torch as pt
from tqdm import tqdm
from glob import glob
import pandas as pd
from uspto.utils_metrics import compute_all_statistics
from uspto.utils_data import (
    get_rdkit_mol,
    write_xyz_file,
    create_bond_graph,
    atom_type_config,
    save_pickle,
    load_pickle,
    Statistics,
)
import tempfile
from uspto.chemformer_encoding import ChemformerEmbedding
from rdkit import Chem
from torch_geometric.data.collate import collate
from torch_geometric.data.separate import separate
from torch_geometric.data import Data, Dataset
import torch
import random
import ase.units as units
import re


class USPTO(Dataset):
    atomic_numbers = [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 34, 35, 50, 53]
    convert_z_to_x = {k: i for i, k in enumerate(atomic_numbers)}

    @property
    def raw_file_names(self):
        return f"", "uspto_dataset.csv"

    @property
    def processed_file_names(self):
        return [
            f"{self.name}.pt",
            f"{self.name}_atom_types.npy",
            f"{self.name}_bond_types.npy",
            f"{self.name}_charge_types.npy",
            f"{self.name}_bond_angles.npy",
            f"{self.name}_num_nodes.pickle",
            f"{self.name}_valencies.pickle",
            f"{self.name}_bond_lengths.pickle",
            f"{self.name}_smiles.pickle",
        ]

    def __init__(
        self,
        root=None,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.name = f"{self.__class__.__name__}"
        self.atom_encoder = atom_type_config("uspto")
        self.chemformer_embedding = ChemformerEmbedding(
            model_path="uspto/data/chemformer.ckpt",
            vocab_path="uspto/data/bart_vocab.txt",
        )

        super().__init__(root, transform, pre_transform, pre_filter)

        self.data, self.slices = pt.load(self.processed_paths[0])

        self.statistics = Statistics(
            atom_types=pt.from_numpy(np.load(self.processed_paths[1])).float(),
            bond_types=pt.from_numpy(np.load(self.processed_paths[2])).float(),
            charge_types=pt.from_numpy(np.load(self.processed_paths[3])).float(),
            bond_angles=pt.from_numpy(np.load(self.processed_paths[4])).float(),
            num_nodes=load_pickle(self.processed_paths[5]),
            valencies=load_pickle(self.processed_paths[6]),
            bond_lengths=load_pickle(self.processed_paths[7]),
        )
        self.smiles = load_pickle(self.processed_paths[8])

    def process(self):
        assert len(self.raw_paths) == 2

        xyz_path = self.raw_paths[0]

        xyz_files = glob(os.path.join(xyz_path, "crest_best_conformer/*.xyz"))
        xyz_files.sort(key=lambda f: int(re.sub("\D", "", f)))
        mol_indices = [int(file.split("/")[-1].split("_")[0]) for file in xyz_files]

        csv_path = self.raw_paths[1]
        csv_file = pd.read_csv(csv_path)

        data_list = []
        smiles_list = []
        j = 0
        for i, d in tqdm(csv_file.iterrows()):
            if i in mol_indices:
                tgt_smiles = d["reactants_can"]
                reactants_embed = self.chemformer_embedding.encode_smiles(tgt_smiles)
                inpt_smiles = d["prod_smiles"]

                file = xyz_files[j]
                idx = int("".join(list(filter(str.isdigit, file.split("/")[-1]))))
                assert i == idx
                j += 1

                mol_dict = parse_xtb_xyz(file, select_random=False, conf_per_mol=1)
                y = torch.tensor(mol_dict["energy"], dtype=torch.float32).unsqueeze(0)
                pos = torch.tensor(mol_dict["pos"], dtype=torch.float32)
                z = torch.tensor(mol_dict["z"], dtype=torch.int64)
                x = torch.tensor(
                    [self.convert_z_to_x[int(a)] for a in z], dtype=pt.long
                )

                with tempfile.NamedTemporaryFile() as tmp:
                    tmp_file = tmp.name
                    # Write xyz file
                    write_xyz_file(coords=pos, atom_types=z, filename=tmp_file)
                    rdkit_mol = get_rdkit_mol(tmp_file)
                    smiles = Chem.MolToSmiles(rdkit_mol, isomericSmiles=False)

                data = Data(
                    z=z,
                    x=x,
                    pos=pos,
                    y=y,
                    mol=rdkit_mol,
                    smiles=inpt_smiles,
                    reactants_embed=reactants_embed,
                )
                data = create_bond_graph(
                    data=data,
                    atom_encoder=self.atom_encoder,
                )
                data_list.append(data)
                smiles_list.append(smiles)

        data, slices = self._collate(data_list)
        pt.save((data, slices), self.processed_paths[0])

        statistics = compute_all_statistics(
            data_list, self.atom_encoder, charges_dic={-1: 0, 0: 1, 1: 2}
        )

        np.save(self.processed_paths[1], statistics.atom_types)
        np.save(self.processed_paths[2], statistics.bond_types)
        np.save(self.processed_paths[3], statistics.charge_types)
        np.save(self.processed_paths[4], statistics.bond_angles)
        save_pickle(statistics.num_nodes, self.processed_paths[5])
        save_pickle(statistics.valencies, self.processed_paths[6])
        save_pickle(statistics.bond_lengths, self.processed_paths[7])
        save_pickle(set(smiles_list), self.processed_paths[8])

    def len(self):
        return len(self.data.y)

    def get(self, idx: int) -> Data:
        data = separate(
            cls=self.data.__class__,
            batch=self.data,
            idx=idx,
            slice_dict=self.slices,
            decrement=False,
        )

        return data

    def _collate(self, data_list):
        if len(data_list) == 1:
            return data_list[0], None

        data, slices, _ = collate(
            data_list[0].__class__,
            data_list=data_list,
            increment=False,
            add_batch=False,
        )
        return data, slices


def parse_xtb_xyz(filename, select_random=False, conf_per_mol=None):
    atom_encoder = {
        "H": 0,
        "Li": 1,
        "B": 2,
        "C": 3,
        "N": 4,
        "O": 5,
        "F": 6,
        "Na": 7,
        "Al": 8,
        "Si": 9,
        "P": 10,
        "S": 11,
        "Cl": 12,
        "K": 13,
        "Ca": 14,
        "Ti": 15,
        "V": 16,
        "Cr": 17,
        "Mn": 18,
        "Fe": 19,
        "Co": 20,
        "Ni": 21,
        "Cu": 22,
        "Zn": 23,
        "Ge": 24,
        "As": 25,
        "Se": 26,
        "Br": 27,
        "Zr": 28,
        "Mo": 29,
        "Pd": 30,
        "Ag": 31,
        "Cd": 32,
        "In": 33,
        "Sn": 34,
        "Sb": 35,
        "I": 36,
        "Ba": 37,
        "Nd": 38,
        "Gd": 39,
        "Yb": 40,
        "Pt": 41,
        "Au": 42,
        "Hg": 43,
        "Pb": 44,
        "Bi": 45,
    }
    atomic_nb = [
        1,
        3,
        5,
        6,
        7,
        8,
        9,
        11,
        13,
        14,
        15,
        16,
        17,
        19,
        20,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        32,
        33,
        34,
        35,
        40,
        42,
        46,
        47,
        48,
        49,
        50,
        51,
        53,
        56,
        60,
        64,
        70,
        78,
        79,
        80,
        82,
        83,
    ]

    if conf_per_mol > 1:
        select_by_energy = True
    else:
        select_by_energy = False
    num_atoms = 0
    atom_type = []
    energy = 0
    pos = []
    with open(filename, "r") as f:
        if select_random or select_by_energy:
            lines = f.readlines()
            num_atoms_total = int(lines[0])
            num_confs = len(lines) // (num_atoms_total + 2)

            if select_random:
                idx = random.randint(0, num_confs - 1)
                random_conf = [
                    (num_atoms_total + 2) * idx,
                    (num_atoms_total + 2) * idx + num_atoms_total + 2,
                ]
                lines = lines[random_conf[0] : random_conf[1]]
                # print(f"Crest conformer with id {idx} chosen")

                for line_num, line in enumerate(lines):
                    if line_num == 0:
                        num_atoms = int(line)
                    elif line_num == 1:
                        # xTB outputs energy in Hartrees: Hartree to eV
                        try:
                            energy = np.array(
                                float(line.split(" ")[-1]) * units.Hartree,
                                dtype=np.float32,
                            )
                        except:
                            energy = np.array(
                                float(line.split(" ")[2]) * units.Hartree,
                                dtype=np.float32,
                            )
                    elif line_num >= 2:
                        t, x, y, z = line.split()
                        try:
                            atom_type.append(atomic_nb[atom_encoder[t]])
                        except:
                            t = t[0] + t[1].lower()
                            atom_type.append(atomic_nb[atom_encoder[t]])
                        pos.append([parse_float(x), parse_float(y), parse_float(z)])

            else:
                energies = dict()
                idx = 0
                num_atoms = 0
                for line_num, line in enumerate(lines):
                    if line_num == 1 + num_atoms:
                        # xTB outputs energy in Hartrees: Hartree to eV
                        try:
                            energy = np.array(
                                float(line.split(" ")[-1]) * units.Hartree,
                                dtype=np.float32,
                            )
                            energies[idx] = energy
                        except:
                            energy = np.array(
                                float(line.split(" ")[2]) * units.Hartree,
                                dtype=np.float32,
                            )
                            energies[idx] = energy
                        idx += 1
                        num_atoms += num_atoms_total + 2
                assert len(energies) == num_confs

                max_confs = conf_per_mol if num_confs > conf_per_mol else num_confs
                energies = sorted(energies.items(), key=lambda x: abs(x[1]))[:max_confs]
                ids = [e[0] for e in energies]
                ranges = [
                    [
                        (num_atoms_total + 2) * i,
                        (num_atoms_total + 2) * i + num_atoms_total + 2,
                    ]
                    for i in ids
                ]
                selected_lines = [
                    lines[ranges[i][0] : ranges[i][1]] for i in range(len(ranges))
                ]
                lines = selected_lines

                energies = []
                atom_types = []
                positions = []
                for conf in lines:
                    atom_type = []
                    pos = []
                    for line_num, line in enumerate(conf):
                        if line_num == 0:
                            num_atoms = int(line)
                        elif line_num == 1:
                            # xTB outputs energy in Hartrees: Hartree to eV
                            try:
                                energy = np.array(
                                    float(line.split(" ")[-1]) * units.Hartree,
                                    dtype=np.float32,
                                )
                            except:
                                energy = np.array(
                                    float(line.split(" ")[2]) * units.Hartree,
                                    dtype=np.float32,
                                )
                        elif line_num >= 2:
                            t, x, y, z = line.split()
                            try:
                                atom_type.append(atomic_nb[atom_encoder[t]])
                            except:
                                t = t[0] + t[1].lower()
                                atom_type.append(atomic_nb[atom_encoder[t]])
                            pos.append([parse_float(x), parse_float(y), parse_float(z)])

                    assert np.array(pos, dtype=np.float32).shape[0] == num_atoms
                    assert np.array(atom_type, dtype=np.int64).shape[0] == num_atoms

                    energies.append(energy)
                    atom_types.append(np.array(atom_type, dtype=np.int64))
                    positions.append(np.array(pos, dtype=np.float32))

                # assert num_atoms_total == num_atoms
                result = {
                    "num_atoms": num_atoms,
                    "z": atom_types,
                    "energy": energies,
                    "pos": positions,
                }
                return result

        else:
            for line_num, line in enumerate(f):
                if line_num == 0:
                    num_atoms = int(line)
                elif line_num == 1:
                    # xTB outputs energy in Hartrees: Hartree to eV
                    try:
                        energy = np.array(
                            float(line.split(" ")[-1]) * units.Hartree, dtype=np.float32
                        )
                    except:
                        energy = np.array(
                            float(line.split(" ")[2]) * units.Hartree, dtype=np.float32
                        )
                elif line_num >= 2:
                    t, x, y, z = line.split()
                    try:
                        atom_type.append(atomic_nb[atom_encoder[t]])
                    except:
                        t = t[0] + t[1].lower()
                        atom_type.append(atomic_nb[atom_encoder[t]])
                    pos.append([parse_float(x), parse_float(y), parse_float(z)])

    # assert num_atoms_total == num_atoms
    num_atoms = num_atoms if not select_by_energy else num_atoms * conf_per_mol
    assert np.array(pos, dtype=np.float32).shape[0] == num_atoms
    assert np.array(atom_type, dtype=np.int64).shape[0] == num_atoms
    result = {
        "num_atoms": num_atoms,
        "z": np.array(atom_type, dtype=np.int64),
        "energy": energy,
        "pos": np.array(pos, dtype=np.float32),
    }
    return result


def parse_float(s: str) -> float:
    try:
        return float(s)
    except ValueError:
        base, power = s.split("*^")
        return float(base) * 10 ** float(power)
