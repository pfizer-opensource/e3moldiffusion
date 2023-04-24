import h5py
import numpy as np
import os
import torch as pt
from tqdm import tqdm

from aqm.utils_metrics import compute_all_statistics
from aqm.utils_data import (
    get_rdkit_mol,
    write_xyz_file,
    create_bond_graph,
    atom_type_config,
    save_pickle,
    load_pickle,
    Statistics,
)
import tempfile
from rdkit import Chem
from torch_geometric.data.collate import collate
from torch_geometric.data.separate import separate
from torch_geometric.data import Data, Dataset


class AQM(Dataset):

    mol_properties = [
        "DIP",
        "HLgap",
        "eAT",
        "eC",
        "eEE",
        "eH",
        "eKIN",
        "eKSE",
        "eL",
        "eNE",
        "eNN",
        "eMBD",
        "eTS",
        "eX",
        "eXC",
        "eXX",
        "mPOL",
    ]

    atomic_energies_dict = {
        1: -13.643321054,
        6: -1027.610746263,
        7: -1484.276217092,
        8: -2039.751675679,
        9: -3139.751675679,
        15: -9283.015861995,
        16: -10828.726222083,
        17: -12516.462339357,
    }
    atomic_numbers = [1, 6, 7, 8, 9, 15, 16, 17]
    convert_z_to_x = {k: i for i, k in enumerate(atomic_numbers)}

    @property
    def raw_file_names(self):
        return "AQM.hdf5"

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
        self.label2idx = {k: i for i, k in enumerate(self.mol_properties)}
        self.atom_encoder = atom_type_config("aqm")

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
        assert len(self.raw_paths) == 1

        data_list = []
        all_smiles = []
        for mol in tqdm(h5py.File(self.raw_paths[0]).values(), desc="Molecules"):

            for group_name in mol:
                group = mol[group_name]

                all_prop = []
                z = pt.tensor(np.array(group["atNUM"]), dtype=pt.long)
                x = pt.tensor([self.convert_z_to_x[int(a)] for a in z], dtype=pt.long)
                all_pos = pt.tensor(np.array(group["atXYZ"]), dtype=pt.float32)
                all_dy = pt.tensor(np.array(group["totFOR"]), dtype=pt.float32)
                for prop in self.mol_properties:
                    tmp = pt.tensor(np.array(group[prop]), dtype=pt.float32)
                    all_prop.append(tmp)
                y = pt.cat(all_prop).unsqueeze(0)

                assert all_pos.shape[0] == all_dy.shape[0]
                assert all_pos.shape[0] == z.shape[0]
                assert all_pos.shape[1] == 3

                assert all_dy.shape[0] == z.shape[0]
                assert all_dy.shape[1] == 3
                assert y.shape[0] == 1
                assert y.shape[1] == len(self.mol_properties)

                with tempfile.NamedTemporaryFile() as tmp:
                    tmp_file = tmp.name
                    # Write xyz file
                    write_xyz_file(coords=all_pos, atom_types=z, filename=tmp_file)
                    rdkit_mol = get_rdkit_mol(tmp_file)
                    smiles = Chem.MolToSmiles(rdkit_mol, isomericSmiles=False)

                data = Data(
                    z=z, x=x, pos=all_pos, y=y, dy=all_dy, mol=rdkit_mol, smiles=smiles
                )
                data = create_bond_graph(
                    data=data,
                    atom_encoder=self.atom_encoder,
                )

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                all_smiles.append(smiles)
                data_list.append(data)

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
        save_pickle(set(all_smiles), self.processed_paths[8])

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

    def get_atomref(self, max_z=10):
        max_z = max(self.atomic_numbers) + 1
        out = pt.zeros(max_z)
        out[list(self.atomic_energies_dict.keys())] = pt.tensor(
            list(self.atomic_energies_dict.values())
        )
        return out.view(-1, 1)

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

    # def __init__(self) -> None:
    # idx_name, z_name, pos_name, y_name, dy_name = self.processed_paths
    # self.idx_mm = np.memmap(idx_name, mode="r", dtype=np.int64)
    # self.z_mm = np.memmap(z_name, mode="r", dtype=np.int8)
    # self.pos_mm = np.memmap(
    #     pos_name, mode="r", dtype=np.float32, shape=(self.z_mm.shape[0], 3)
    # )
    # self.y_mm = np.memmap(
    #     y_name,
    #     mode="r",
    #     dtype=np.float64,
    #     shape=(len(self.idx_mm) - 1, len(self.mol_properties)),
    # )
    # self.dy_mm = np.memmap(
    #     dy_name, mode="r", dtype=np.float32, shape=(self.z_mm.shape[0], 3)
    # )

    # assert self.idx_mm[0] == 0
    # assert self.idx_mm[-1] == len(self.z_mm)
    # assert len(self.idx_mm) == len(self.y_mm) + 1

    # def process(self):
    #     print("Gathering statistics...")
    #     num_all_confs = 0
    #     num_all_atoms = 0
    #     for data in self.sample_iter():
    #         num_all_confs += 1
    #         num_all_atoms += data.z.shape[0]

    #     print(f"  Total number of conformers: {num_all_confs}")
    #     print(f"  Total number of atoms: {num_all_atoms}")

    #     idx_name, z_name, pos_name, y_name, dy_name = self.processed_paths
    #     idx_mm = np.memmap(
    #         idx_name + ".tmp", mode="w+", dtype=np.int64, shape=(num_all_confs + 1,)
    #     )
    #     z_mm = np.memmap(
    #         z_name + ".tmp", mode="w+", dtype=np.int8, shape=(num_all_atoms,)
    #     )
    #     pos_mm = np.memmap(
    #         pos_name + ".tmp", mode="w+", dtype=np.float32, shape=(num_all_atoms, 3)
    #     )
    #     y_mm = np.memmap(
    #         y_name + ".tmp",
    #         mode="w+",
    #         dtype=np.float64,
    #         shape=(num_all_confs, len(self.mol_properties)),
    #     )
    #     dy_mm = np.memmap(
    #         dy_name + ".tmp", mode="w+", dtype=np.float32, shape=(num_all_atoms, 3)
    #     )

    #     print("Storing data...")
    #     i_atom = 0
    #     for i_conf, data in enumerate(self.sample_iter()):
    #         i_next_atom = i_atom + data.z.shape[0]

    #         idx_mm[i_conf] = i_atom
    #         z_mm[i_atom:i_next_atom] = data.z.to(pt.int8)
    #         pos_mm[i_atom:i_next_atom] = data.pos
    #         y_mm[i_conf, :] = data.y
    #         dy_mm[i_atom:i_next_atom] = data.dy

    #         i_atom = i_next_atom

    #     idx_mm[-1] = num_all_atoms
    #     assert i_atom == num_all_atoms

    #     idx_mm.flush()
    #     z_mm.flush()
    #     pos_mm.flush()
    #     y_mm.flush()
    #     dy_mm.flush()

    #     os.rename(idx_mm.filename, idx_name)
    #     os.rename(z_mm.filename, z_name)
    #     os.rename(pos_mm.filename, pos_name)
    #     os.rename(y_mm.filename, y_name)
    #     os.rename(dy_mm.filename, dy_name)

    # def get(self, idx):

    #     atoms = slice(self.idx_mm[idx], self.idx_mm[idx + 1])
    #     z = pt.tensor(self.z_mm[atoms], dtype=pt.long)
    #     pos = pt.tensor(self.pos_mm[atoms], dtype=pt.float32)
    #     y = pt.tensor(self.y_mm[idx], dtype=pt.float32).view(1, -1)
    #     dy = pt.tensor(self.dy_mm[atoms], dtype=pt.float32)
    #     return Data(z=z, pos=pos, y=y, dy=dy)

    # def len(self):
    #     return len(self.y_mm)
