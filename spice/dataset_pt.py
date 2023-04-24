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
        version="1.1.3",
        subsets=None,
        max_gradient=None,
        subsample_molecules=1,
    ):
        self.atom_encoder = atom_type_config("spice")

        arg_hash = f"{version}{subsets}{max_gradient}{subsample_molecules}"
        arg_hash = hashlib.md5(arg_hash.encode()).hexdigest()
        self.name = f"{self.__class__.__name__}-{arg_hash}"
        self.version = str(version)
        assert self.version in self.VERSIONS
        self.subsets = subsets
        self.max_gradient = max_gradient
        self.subsample_molecules = int(subsample_molecules)
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
        assert self.subsample_molecules > 0

        data_list = []
        all_smiles = []

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
                data = Data(
                    z=z, x=x, pos=pos, y=y, dy=neg_dy, mol=rdkit_mol, smiles=smiles
                )
                data = create_bond_graph(
                    data=data,
                    atom_encoder=self.atom_encoder,
                )

                all_smiles.append(smiles)
                data_list.append(data)

        save_pickle(set(all_smiles), self.processed_paths[8])
        del all_smiles

        data, slices = self._collate(data_list)
        pt.save((data, slices), self.processed_paths[0])
        statistics = compute_all_statistics(
            data_list,
            self.atom_encoder,
            charges_dic={-2: 0, -1: 1, 0: 2, 1: 3, 2: 4, 3: 5},
        )
        del data_list

        np.save(self.processed_paths[1], statistics.atom_types)
        np.save(self.processed_paths[2], statistics.bond_types)
        np.save(self.processed_paths[3], statistics.charge_types)
        np.save(self.processed_paths[4], statistics.bond_angles)
        save_pickle(statistics.num_nodes, self.processed_paths[5])
        save_pickle(statistics.valencies, self.processed_paths[6])
        save_pickle(statistics.bond_lengths, self.processed_paths[7])

    def download(self):
        download_url(self.raw_url, self.raw_dir)

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
