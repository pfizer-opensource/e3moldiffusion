import torch
from torch_geometric.transforms import Compose
from torch_geometric.datasets import QM9 as QM9_geometric
from torch_geometric.nn.models.schnet import qm9_target_dict
from tqdm import tqdm
import numpy as np
from torch_geometric.data import (
    Data,
)
from torch_geometric.utils import scatter

from qm9.utils_metrics import compute_all_statistics
from qm9.utils_data import (
    save_pickle,
    load_pickle,
    Statistics,
    atom_type_config,
)
from qm9.utils import one_hot

HAR2EV = 27.211386246
KCALMOL2EV = 0.04336414

conversion = torch.tensor(
    [
        1.0,
        1.0,
        HAR2EV,
        HAR2EV,
        HAR2EV,
        1.0,
        HAR2EV,
        HAR2EV,
        HAR2EV,
        HAR2EV,
        HAR2EV,
        1.0,
        KCALMOL2EV,
        KCALMOL2EV,
        KCALMOL2EV,
        KCALMOL2EV,
        1.0,
        1.0,
        1.0,
    ]
)

LABELS = [
    "dipole_moment",
    "isotropic_polarizability",
    "homo",
    "lumo",
    "gap",
    "electronic_spatial_extent",
    "zpve",
    "energy_U0",
    "energy_U",
    "enthalpy_H",
    "free_energy",
    "heat_capacity",
]
LABELS_IDX = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]


class QM9(QM9_geometric):
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

    atomic_numbers = [1, 6, 7, 8, 9]
    convert_z_to_x = {k: i for i, k in enumerate(atomic_numbers)}

    def __init__(self, root, transform=None, label=None):
        self.name = f"{self.__class__.__name__}"
        self.atom_encoder = atom_type_config("qm9")

        self.label2idx = dict(zip(qm9_target_dict.values(), qm9_target_dict.keys()))

        if isinstance(label, list):
            label_idx = []
            for l in label:
                assert l in self.label2idx, (
                    "Please pass the desired property to "
                    'train on via "label". Available '
                    f'properties are {", ".join(self.label2idx)}.'
                )
                label_idx.append(self.label2idx[l])
            self.label_idx = label_idx

        elif isinstance(label, str):
            assert label in self.label2idx, (
                "Please pass the desired property to "
                'train on via "label". Available '
                f'properties are {", ".join(self.label2idx)}.'
            )
            self.label_idx = self.label2idx[label]

        self.label = label

        super().__init__(root, transform=transform)

        self.data, self.slices = torch.load(self.processed_paths[0])

        self.statistics = Statistics(
            atom_types=torch.from_numpy(np.load(self.processed_paths[1])).float(),
            bond_types=torch.from_numpy(np.load(self.processed_paths[2])).float(),
            charge_types=torch.from_numpy(np.load(self.processed_paths[3])).float(),
            bond_angles=torch.from_numpy(np.load(self.processed_paths[4])).float(),
            num_nodes=load_pickle(self.processed_paths[5]),
            valencies=load_pickle(self.processed_paths[6]),
            bond_lengths=load_pickle(self.processed_paths[7]),
        )
        self.smiles = load_pickle(self.processed_paths[8])

    def get_atomref(self, max_z=100):
        atomref = self.atomref(self.label_idx)
        if atomref is None:
            return None
        if atomref.size(0) != max_z:
            tmp = torch.zeros(max_z).unsqueeze(1)
            idx = min(max_z, atomref.size(0))
            tmp[:idx] = atomref[:idx]
            return tmp
        return atomref

    def _filter_label(self, batch):
        if isinstance(self.label, list):
            for label in self.label:
                target = batch.y[:, self.label2idx[label]].unsqueeze(1)
                batch[label] = target
        else:
            batch.y = batch.y[:, self.label_idx].unsqueeze(1)
        return batch

    def download(self):
        super(QM9, self).download()

    def process(self):
        import rdkit
        from rdkit import Chem, RDLogger
        from rdkit.Chem.rdchem import BondType as BT
        from rdkit.Chem.rdchem import HybridizationType

        RDLogger.DisableLog("rdApp.*")

        types = {"H": 0, "C": 1, "N": 2, "O": 3, "F": 4}
        bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

        with open(self.raw_paths[1], "r") as f:
            target = f.read().split("\n")[1:-1]
            target = [[float(x) for x in line.split(",")[1:20]] for line in target]
            target = torch.tensor(target, dtype=torch.float)
            target = torch.cat([target[:, 3:], target[:, :3]], dim=-1)
            target = target * conversion.view(1, -1)

        with open(self.raw_paths[2], "r") as f:
            skip = [int(x.split()[0]) - 1 for x in f.read().split("\n")[9:-2]]

        suppl = Chem.SDMolSupplier(self.raw_paths[0], removeHs=False, sanitize=False)

        data_list = []
        smiles_list = []
        for i, mol in enumerate(tqdm(suppl)):
            if i in skip:
                continue

            N = mol.GetNumAtoms()

            conf = mol.GetConformer()
            pos = conf.GetPositions()
            pos = torch.tensor(pos, dtype=torch.float)

            smiles = Chem.rdmolfiles.MolToSmiles(mol, isomericSmiles=False)

            type_idx = []
            atomic_number = []
            aromatic = []
            sp = []
            sp2 = []
            sp3 = []
            num_hs = []
            for atom in mol.GetAtoms():
                type_idx.append(types[atom.GetSymbol()])
                atomic_number.append(atom.GetAtomicNum())
                aromatic.append(1 if atom.GetIsAromatic() else 0)
                hybridization = atom.GetHybridization()
                sp.append(1 if hybridization == HybridizationType.SP else 0)
                sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
                sp3.append(1 if hybridization == HybridizationType.SP3 else 0)

            z = torch.tensor(atomic_number, dtype=torch.long)
            x = torch.tensor([self.convert_z_to_x[int(a)] for a in z], dtype=torch.long)
            all_charges = []
            for atom in mol.GetAtoms():
                all_charges.append(atom.GetFormalCharge())
            all_charges = torch.Tensor(all_charges).long()

            row, col, edge_type = [], [], []
            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                row += [start, end]
                col += [end, start]
                edge_type += 2 * [bonds[bond.GetBondType()]]

            edge_index = torch.tensor([row, col], dtype=torch.long)
            edge_attr = torch.tensor(edge_type, dtype=torch.long) + 1

            perm = (edge_index[0] * N + edge_index[1]).argsort()
            edge_index = edge_index[:, perm]
            edge_attr = edge_attr[perm]

            row, col = edge_index
            hs = (z == 1).to(torch.float)
            num_hs = scatter(hs[row], col, dim_size=N, reduce="sum").tolist()

            x1 = one_hot(torch.tensor(type_idx), num_classes=len(types))
            x2 = (
                torch.tensor(
                    [atomic_number, aromatic, sp, sp2, sp3, num_hs], dtype=torch.float
                )
                .t()
                .contiguous()
            )
            atom_features = torch.cat([x1, x2], dim=-1)

            y = target[i].unsqueeze(0)
            name = mol.GetProp("_Name")

            data = Data(
                x=x,
                z=z,
                pos=pos,
                bond_index=edge_index,
                bond_attr=edge_attr,
                y=y,
                charges=all_charges,
                atom_feat=atom_features,
                name=name,
                idx=i,
            )

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)
            smiles_list.append(smiles)

        torch.save(self.collate(data_list), self.processed_paths[0])

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
