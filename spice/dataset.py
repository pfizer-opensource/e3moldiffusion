import hashlib
import h5py
import numpy as np
import os
import torch as pt
from torch_geometric.data import Data, Dataset, download_url
from tqdm import tqdm

from spice.utils_metrics import compute_all_statistics
from spice.utils_data import (
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


class SPICE(Dataset):
    HARTREE_TO_EV = 27.211386246
    BORH_TO_ANGSTROM = 0.529177

    VERSIONS = {
        "1.0": {
            "url": "https://github.com/openmm/spice-dataset/releases/download/1.0",
            "file": "SPICE.hdf5",
        },
        "1.1": {
            "url": "https://github.com/openmm/spice-dataset/releases/download/1.1",
            "file": "SPICE.hdf5",
        },
        "1.1.1": {
            "url": "https://zenodo.org/record/7258940/files",
            "file": "SPICE-1.1.1.hdf5",
        },
        "1.1.2": {
            "url": "https://zenodo.org/record/7338495/files",
            "file": "SPICE-1.1.2.hdf5",
        },
        "1.1.3": {
            "url": "https://zenodo.org/record/7606550/files",
            "file": "SPICE-1.1.3.hdf5",
        },
    }

    atomic_numbers = [1, 3, 6, 7, 8, 9, 11, 12, 15, 16, 17, 19, 20, 35, 53]
    convert_z_to_x = {k: i for i, k in enumerate(atomic_numbers)}

    @property
    def raw_dir(self):
        return os.path.join(super().raw_dir, self.version)

    @property
    def raw_file_names(self):
        return self.VERSIONS[self.version]["file"]

    @property
    def raw_url(self):
        return f"{self.VERSIONS[self.version]['url']}/{self.VERSIONS[self.version]['file']}"

    @property
    def processed_file_names(self):
        return [
            f"{self.name}.idx.mmap",
            f"{self.name}.edge_idx.mmap",
            f"{self.name}.z.mmap",
            f"{self.name}.pos.mmap",
            f"{self.name}.y.mmap",
            f"{self.name}.charges.mmap",
            f"{self.name}.bond_index.mmap",
            f"{self.name}.bond_attr.mmap",
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
        version="1.1.3",
        subsets=None,
        max_gradient=None,
        subsample_molecules=1,
    ):
        self.atom_encoder = atom_type_config("spice")
        self.data_list = []
        self.all_smiles = []

        arg_hash = f"{version}{subsets}{max_gradient}{subsample_molecules}"
        arg_hash = hashlib.md5(arg_hash.encode()).hexdigest()
        self.name = f"{self.__class__.__name__}-{arg_hash}"
        self.version = str(version)
        assert self.version in self.VERSIONS
        self.subsets = subsets
        self.max_gradient = max_gradient
        self.subsample_molecules = int(subsample_molecules)
        super().__init__(root, transform, pre_transform, pre_filter)

        (
            idx_name,
            edge_idx_name,
            z_name,
            pos_name,
            y_name,
            charges_name,
            bond_index_name,
            bond_attr_name,
        ) = self.processed_paths[:8]
        self.idx_mm = np.memmap(idx_name, mode="r", dtype=np.int64)
        self.edge_idx_mm = np.memmap(edge_idx_name, mode="r", dtype=np.int64)
        self.bond_index_mm = np.memmap(
            bond_index_name,
            mode="r",
            dtype=np.int64,
            shape=(2, np.sum(self.edge_idx_mm)),
        )
        self.z_mm = np.memmap(z_name, mode="r", dtype=np.int8)
        self.charges_mm = np.memmap(charges_name, mode="r", dtype=np.int64)
        self.pos_mm = np.memmap(
            pos_name, mode="r", dtype=np.float32, shape=(self.z_mm.shape[0], 3)
        )
        self.bond_attr_mm = np.memmap(
            bond_attr_name,
            mode="r",
            dtype=np.int64,
            shape=(self.bond_index_mm[0].shape[0],),
        )
        self.y_mm = np.memmap(y_name, mode="r", dtype=np.float64)

        self.statistics = Statistics(
            atom_types=pt.from_numpy(np.load(self.processed_paths[8])).float(),
            bond_types=pt.from_numpy(np.load(self.processed_paths[9])).float(),
            charge_types=pt.from_numpy(np.load(self.processed_paths[10])).float(),
            bond_angles=pt.from_numpy(np.load(self.processed_paths[11])).float(),
            num_nodes=load_pickle(self.processed_paths[12]),
            valencies=load_pickle(self.processed_paths[13]),
            bond_lengths=load_pickle(self.processed_paths[14]),
        )
        self.smiles = load_pickle(self.processed_paths[15])

    def sample_iter(self):
        assert len(self.raw_paths) == 1
        assert self.subsample_molecules > 0

        molecules = h5py.File(self.raw_paths[0]).items()
        for i_mol, (mol_id, mol) in tqdm(enumerate(molecules), desc="Molecules"):
            if self.subsets:
                if mol["subset"][0].decode() not in list(self.subsets):
                    continue

            # Subsample molecules
            if i_mol % self.subsample_molecules != 0:
                continue

            z = pt.tensor(mol["atomic_numbers"], dtype=pt.long)
            all_pos = (
                pt.tensor(mol["conformations"], dtype=pt.float32)
                * self.BORH_TO_ANGSTROM
            )
            all_y = (
                pt.tensor(mol["formation_energy"], dtype=pt.float64)
                * self.HARTREE_TO_EV
            )
            all_neg_dy = (
                -pt.tensor(mol["dft_total_gradient"], dtype=pt.float32)
                * self.HARTREE_TO_EV
                / self.BORH_TO_ANGSTROM
            )

            assert all_pos.shape[0] == all_y.shape[0]
            assert all_pos.shape[1] == z.shape[0]
            assert all_pos.shape[2] == 3

            assert all_neg_dy.shape[0] == all_y.shape[0]
            assert all_neg_dy.shape[1] == z.shape[0]
            assert all_neg_dy.shape[2] == 3

            for pos, y, neg_dy in zip(all_pos, all_y, all_neg_dy):
                # Skip samples with large forces
                if self.max_gradient:
                    if neg_dy.norm(dim=1).max() > float(self.max_gradient):
                        continue

                with tempfile.NamedTemporaryFile() as tmp:
                    tmp_file = tmp.name
                    # Write xyz file
                    write_xyz_file(coords=pos, atom_types=z, filename=tmp_file)
                    rdkit_mol = get_rdkit_mol(tmp_file)
                    smiles = Chem.MolToSmiles(rdkit_mol, isomericSmiles=False)

                x = pt.tensor([self.convert_z_to_x[int(a)] for a in z], dtype=pt.long)
                data = Data(z=z, x=x, pos=pos, y=y, mol=rdkit_mol, smiles=smiles)
                data = create_bond_graph(
                    data=data,
                    atom_encoder=self.atom_encoder,
                )

                yield data

    def download(self):
        download_url(self.raw_url, self.raw_dir)

    def process(self):
        print("Arguments")
        print(f"  version: {self.version}")
        print(f"  subsets: {self.subsets}")
        print(f"  max_gradient: {self.max_gradient} eV/A")
        print(f"  subsample_molecules: {self.subsample_molecules}\n")

        print("Gathering statistics...")
        num_all_confs = 0
        num_all_atoms = 0
        num_all_edges = 0
        for data in self.sample_iter():
            num_all_confs += 1
            num_all_atoms += data.z.shape[0]
            num_all_edges += data.bond_index[0].shape[0]

        print(f"  Total number of conformers: {num_all_confs}")
        print(f"  Total number of atoms: {num_all_atoms}")
        print(f"  Total number of edges: {num_all_edges}")

        (
            idx_name,
            edge_idx_name,
            z_name,
            pos_name,
            y_name,
            charges_name,
            bond_index_name,
            bond_attr_name,
        ) = self.processed_paths[:8]
        idx_mm = np.memmap(
            idx_name + ".tmp", mode="w+", dtype=np.int64, shape=(num_all_confs + 1,)
        )
        edge_idx_mm = np.memmap(
            edge_idx_name + ".tmp",
            mode="w+",
            dtype=np.int64,
            shape=(num_all_confs + 1,),
        )
        bond_index_mm = np.memmap(
            bond_index_name + ".tmp",
            mode="w+",
            dtype=np.int64,
            shape=(2, num_all_edges),
        )
        z_mm = np.memmap(
            z_name + ".tmp", mode="w+", dtype=np.int8, shape=(num_all_atoms,)
        )
        charges_mm = np.memmap(
            charges_name + ".tmp", mode="w+", dtype=np.int64, shape=(num_all_atoms,)
        )
        pos_mm = np.memmap(
            pos_name + ".tmp", mode="w+", dtype=np.float32, shape=(num_all_atoms, 3)
        )

        bond_attr_mm = np.memmap(
            bond_attr_name + ".tmp",
            mode="w+",
            dtype=np.int64,
            shape=(num_all_edges,),
        )
        y_mm = np.memmap(
            y_name + ".tmp", mode="w+", dtype=np.float64, shape=(num_all_confs,)
        )

        data_list = []
        print("Storing data...")
        i_atom = 0
        i_edge = 0
        for i_conf, data in enumerate(self.sample_iter()):
            i_next_atom = i_atom + data.z.shape[0]
            i_next_edge = i_edge + data.bond_index[0].shape[0]

            idx_mm[i_conf] = i_atom
            edge_idx_mm[i_conf] = i_edge
            bond_index_mm[:, i_edge:i_next_edge] = data.bond_index
            z_mm[i_atom:i_next_atom] = data.z.to(pt.int8)
            charges_mm[i_atom:i_next_atom] = data.charges.to(pt.int64)
            pos_mm[i_atom:i_next_atom] = data.pos
            bond_attr_mm[i_edge:i_next_edge] = data.bond_attr
            y_mm[i_conf] = data.y

            i_atom = i_next_atom
            i_edge = i_next_edge

            data_list.append(data)

        idx_mm[-1] = num_all_atoms
        assert i_atom == num_all_atoms
        assert i_edge == num_all_edges

        idx_mm.flush()
        edge_idx_mm.flush()
        z_mm.flush()
        pos_mm.flush()
        y_mm.flush()
        bond_index_mm.flush()
        bond_attr_mm.flush()
        charges_mm.flush()

        os.rename(idx_mm.filename, idx_name)
        os.rename(edge_idx_mm.filename, edge_idx_name)
        os.rename(z_mm.filename, z_name)
        os.rename(pos_mm.filename, pos_name)
        os.rename(y_mm.filename, y_name)
        os.rename(charges_mm.filename, charges_name)
        os.rename(bond_index_mm.filename, bond_index_name)
        os.rename(bond_attr_mm.filename, bond_attr_name)

        statistics = compute_all_statistics(
            data_list,
            self.atom_encoder,
            charges_dic={-2: 0, -1: 1, 0: 2, 1: 3, 2: 4, 3: 5},
        )
        all_smiles = [data.smiles for data in data_list]
        del data_list

        np.save(self.processed_paths[8], statistics.atom_types)
        np.save(self.processed_paths[9], statistics.bond_types)
        np.save(self.processed_paths[10], statistics.charge_types)
        np.save(self.processed_paths[11], statistics.bond_angles)
        save_pickle(statistics.num_nodes, self.processed_paths[12])
        save_pickle(statistics.valencies, self.processed_paths[13])
        save_pickle(statistics.bond_lengths, self.processed_paths[14])

        save_pickle(set(all_smiles), self.processed_paths[15])

    def len(self):
        return len(self.y_mm)

    def get(self, idx):
        atoms = slice(self.idx_mm[idx], self.idx_mm[idx + 1])
        edges = slice(self.edge_idx_mm[idx], self.edge_idx_mm[idx + 1])

        z = pt.tensor(self.z_mm[atoms], dtype=pt.long)
        charges = pt.tensor(self.charges_mm[atoms], dtype=pt.long)
        pos = pt.tensor(self.pos_mm[atoms], dtype=pt.float32)
        bond_attr = pt.tensor(self.bond_attr_mm[edges], dtype=pt.long)
        bond_index = pt.tensor(self.bond_index_mm[:, edges], dtype=pt.long)
        y = pt.tensor(self.y_mm[idx], dtype=pt.float32).view(1, 1)

        return Data(
            z=z,
            pos=pos,
            y=y,
            charges=charges,
            bond_index=bond_index,
            bond_attr=bond_attr,
        )
