import torch
import torch.nn.functional as F
from experiments.molecule_utils import PlaceHolder
from experiments.data.abstract_dataset import (
    AbstractDatasetInfos,
)

full_atom_encoder_drugs = {'H': 0, 'B': 1, 'C': 2, 'N': 3, 'O': 4, 'F': 5, 'Al': 6, 'Si': 7,
                     'P': 8, 'S': 9, 'Cl': 10, 'As': 11, 'Br': 12, 'I': 13, 'Hg': 14, 'Bi': 15}
class GEOMInfos(AbstractDatasetInfos):
    def __init__(self, datamodule, cfg):
        self.remove_h = cfg.remove_hs
        self.need_to_strip = False        # to indicate whether we need to ignore one output from the model
        self.statistics = datamodule.statistics
        self.name = 'drugs'
        self.atom_encoder = full_atom_encoder_drugs
        self.charge_offset = 2
        self.collapse_charges = torch.Tensor([-2, -1, 0, 1, 2, 3]).int()
        if self.remove_h:
            self.atom_encoder = {k: v - 1 for k, v in self.atom_encoder.items() if k != 'H'}

        super().complete_infos(datamodule.statistics, self.atom_encoder)

        self.input_dims = PlaceHolder(X=self.num_atom_types, C=6, E=5, y=1, pos=3)
        self.output_dims = PlaceHolder(X=self.num_atom_types, C=6, E=5, y=0, pos=3)

    def to_one_hot(self, X, C, E, node_mask):
        X = F.one_hot(X, num_classes=self.num_atom_types).float()
        E = F.one_hot(E, num_classes=5).float()
        C = F.one_hot(C + self.charge_offset, num_classes=6).float()
        placeholder = PlaceHolder(X=X, C=C, E=E, y=None, pos=None)
        pl = placeholder.mask(node_mask)
        return pl.X, pl.C, pl.E

    def one_hot_charges(self, C):
        return F.one_hot((C + self.charge_offset).long(), num_classes=6).float()
    

full_atom_encoder_qm9 = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
class QM9Infos(AbstractDatasetInfos):
    def __init__(self, datamodule, cfg):
        self.remove_h = cfg.remove_hs
        self.statistics = datamodule.statistics
        self.name = 'qm9'
        self.atom_encoder = full_atom_encoder_qm9
        self.charge_offset = 1
        self.collapse_charges = torch.Tensor([-1, 0, 1]).int()
        if self.remove_h:
            self.atom_encoder = {k: v - 1 for k, v in self.atom_encoder.items() if k != 'H'}
        super().complete_infos(datamodule.statistics, self.atom_encoder)
        self.input_dims = PlaceHolder(X=self.num_atom_types, C=3, E=5, y=1, pos=3)
        self.output_dims = PlaceHolder(X=self.num_atom_types, C=3, E=5, y=0, pos=3)

    def to_one_hot(self, X, C, E, node_mask):
        X = F.one_hot(X, num_classes=self.num_atom_types).float()
        E = F.one_hot(E, num_classes=5).float()
        C = F.one_hot(C + self.charge_offset, num_classes=3).float()
        placeholder = PlaceHolder(X=X, C=C, E=E,  y=None, pos=None)
        pl = placeholder.mask(node_mask)
        return pl.X, pl.C, pl.E

    def one_hot_charges(self, charges):
        return F.one_hot((charges + self.charge_offset).long(), num_classes=3).float()