from torch_geometric.data import Data
from rdkit import Chem, RDLogger
from openbabel import pybel
from rdkit import Chem
from openbabel import pybel
from rdkit import Chem
import torch
import torch.nn.functional as F

import torch


class DistributionNodes:
    def __init__(self, histogram):
        """Compute the distribution of the number of nodes in the dataset, and sample from this distribution.
        historgram: dict. The keys are num_nodes, the values are counts
        """

        if type(histogram) == dict:
            max_n_nodes = max(histogram.keys())
            prob = torch.zeros(max_n_nodes + 1)
            for num_nodes, count in histogram.items():
                prob[num_nodes] = count
        else:
            prob = histogram

        self.prob = prob / prob.sum()
        self.m = torch.distributions.Categorical(prob)

    def sample_n(self, n_samples, device):
        idx = self.m.sample((n_samples,))
        return idx.to(device)

    def log_prob(self, batch_n_nodes):
        assert len(batch_n_nodes.size()) == 1
        probas = self.prob[batch_n_nodes.to(self.prob.device)]
        log_p = torch.log(probas + 1e-10)
        return log_p.to(batch_n_nodes.device)


class PlaceHolder:
    def __init__(self, pos, X, charges, E, y, t_int=None, t=None):
        self.pos = pos
        self.X = X
        self.charges = charges
        self.E = E
        self.y = y
        self.t_int = t_int
        self.t = t


class AbstractDatasetInfos:
    def complete_infos(self, statistics, atom_encoder):
        self.atom_decoder = [key for key in atom_encoder.keys()]
        self.num_atom_types = len(self.atom_decoder)

        all_nodes = statistics.num_nodes
        max_n_nodes = max(all_nodes.keys())
        n_nodes = torch.zeros(max_n_nodes + 1, dtype=torch.long)
        for key, value in all_nodes.items():
            n_nodes[key] += value

        self.n_nodes = n_nodes / n_nodes.sum()
        self.atom_types = statistics.atom_types
        self.edge_types = statistics.bond_types
        self.charges_types = statistics.charge_types
        self.charges_marginals = (self.charges_types * self.atom_types[:, None]).sum(
            dim=0
        )
        self.valency_distribution = statistics.valencies
        self.max_n_nodes = len(n_nodes) - 1
        self.nodes_dist = DistributionNodes(n_nodes)


class SPICEInfos(AbstractDatasetInfos):
    def __init__(self, datamodule):
        self.name = "aqm"

        self.statistics = datamodule.statistics
        self.atom_encoder = datamodule.atom_encoder
        self.collapse_charges = torch.Tensor([-1, 0, 1]).int()

        super().complete_infos(datamodule.statistics, self.atom_encoder)

        self.input_dims = PlaceHolder(X=self.num_atom_types, charges=3, E=5, y=1, pos=3)
        self.output_dims = PlaceHolder(
            X=self.num_atom_types, charges=3, E=5, y=0, pos=3
        )

    def to_one_hot(self, X, charges, E, node_mask):
        X = F.one_hot(X, num_classes=self.num_atom_types).float()
        E = F.one_hot(E, num_classes=5).float()
        charges = F.one_hot(charges + 1, num_classes=3).float()
        placeholder = PlaceHolder(X=X, charges=charges, E=E, y=None, pos=None)
        pl = placeholder.mask(node_mask)
        return pl.X, pl.charges, pl.E

    def one_hot_charges(self, charges):
        return F.one_hot((charges + 1).long(), num_classes=3).float()
