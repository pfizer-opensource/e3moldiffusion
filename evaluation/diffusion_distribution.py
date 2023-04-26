from torch.distributions.categorical import Categorical
import numpy as np
import torch

PROP_TO_IDX = {
    "DIP": 0,
    "HLgap": 1,
    "eAT": 2,
    "eC": 3,
    "eEE": 4,
    "eH": 5,
    "eKIN": 6,
    "eKSE": 7,
    "eL": 8,
    "eNE": 9,
    "eNN": 10,
    "eMBD": 11,
    "eTS": 12,
    "eX": 13,
    "eXC": 14,
    "eXX": 15,
    "mPOL": 16,
}

IDX_TO_PROP = {v: k for k, v in PROP_TO_IDX.items()}


def get_distributions(args, dataset_info, datamodule):
    histogram = dataset_info["n_nodes"]
    in_node_nf = len(dataset_info["atom_decoder"]) + int(args.include_charges)
    nodes_dist = DistributionNodes(histogram)

    prop_dist = None
    if len(args.properties_list) > 0:
        prop_dist = DistributionProperty(datamodule, args.properties_list)

    return nodes_dist, prop_dist


class DistributionNodes:
    def __init__(self, histogram):

        self.n_nodes = []
        prob = []
        self.keys = {}
        for i, nodes in enumerate(histogram):
            self.n_nodes.append(nodes)
            self.keys[nodes] = i
            prob.append(histogram[nodes])
        self.n_nodes = torch.tensor(self.n_nodes)
        prob = np.array(prob)
        prob = prob / np.sum(prob)

        self.prob = torch.from_numpy(prob).float()

        entropy = torch.sum(self.prob * torch.log(self.prob + 1e-30))
        print("Entropy of n_nodes: H[N]", entropy.item())

        self.m = Categorical(torch.tensor(prob))

    def sample(self, n_samples=1):
        idx = self.m.sample((n_samples,))
        return self.n_nodes[idx]

    def log_prob(self, batch_n_nodes):
        assert len(batch_n_nodes.size()) == 1

        idcs = [self.keys[i.item()] for i in batch_n_nodes]
        idcs = torch.tensor(idcs).to(batch_n_nodes.device)

        log_p = torch.log(self.prob + 1e-30)

        log_p = log_p.to(batch_n_nodes.device)

        log_probs = log_p[idcs]

        return log_probs


class DistributionProperty:
    def __init__(self, dataloader, properties, num_bins=1000, normalizer=None):
        self.num_bins = num_bins
        self.distributions = {}
        self.properties = properties

        # train_idx = [int(i) for i in datamodule.idx_train]
        # mols = datamodule.dataset.data.mol
        # num_atoms = torch.tensor(
        #     [a.GetNumAtoms() for i, a in enumerate(mols) if i in train_idx]
        # )
        num_atoms = torch.tensor([len(data.z) for data in dataloader.dataset[:]])
        for prop in properties:
            self.distributions[prop] = {}

            # idx = datamodule.dataset.label2idx[prop]
            # property = datamodule.dataset.data.y[:, idx]
            # property = torch.tensor(
            #     [a for i, a in enumerate(property) if i in train_idx]
            # )
            idx = dataloader.dataset[:].label2idx[prop]
            property = torch.tensor([data.y[:, idx] for data in dataloader.dataset[:]])
            self._create_prob_dist(num_atoms, property, self.distributions[prop])

        self.normalizer = normalizer

    def set_normalizer(self, normalizer):
        self.normalizer = normalizer

    def _create_prob_dist(self, nodes_arr, values, distribution):
        min_nodes, max_nodes = torch.min(nodes_arr), torch.max(nodes_arr)
        for n_nodes in range(int(min_nodes), int(max_nodes) + 1):
            idxs = nodes_arr == n_nodes
            values_filtered = values[idxs]
            if len(values_filtered) > 0:
                probs, params = self._create_prob_given_nodes(values_filtered)
                distribution[n_nodes] = {"probs": probs, "params": params}

    def _create_prob_given_nodes(self, values):
        n_bins = self.num_bins  # min(self.num_bins, len(values))
        prop_min, prop_max = torch.min(values), torch.max(values)
        prop_range = prop_max - prop_min + 1e-12
        histogram = torch.zeros(n_bins)
        for val in values:
            i = int((val - prop_min) / prop_range * n_bins)
            # Because of numerical precision, one sample can fall in bin int(n_bins) instead of int(n_bins-1)
            # We move it to bin int(n_bind-1 if that happens)
            if i == n_bins:
                i = n_bins - 1
            histogram[i] += 1
        probs = histogram / torch.sum(histogram)
        try:
            probs = Categorical(probs)
        except:
            probs = Categorical(torch.tensor(probs))
        params = [prop_min, prop_max]
        return probs, params

    def normalize_tensor(self, tensor, prop):
        assert self.normalizer is not None
        mean = self.normalizer[prop]["mean"]
        mad = self.normalizer[prop]["mad"]
        return (tensor - mean) / mad

    def sample(self, n_nodes=19):
        vals = []
        for prop in self.properties:
            dist = self.distributions[prop][n_nodes]
            idx = dist["probs"].sample((1,))
            val = self._idx2value(idx, dist["params"], len(dist["probs"].probs))
            val = self.normalize_tensor(val, prop)
            vals.append(val)
        vals = torch.cat(vals)
        return vals

    def sample_specific(self, prop, n_nodes=19):
        dist = self.distributions[prop][n_nodes]
        idx = dist["probs"].sample((1,))
        val = self._idx2value(idx, dist["params"], len(dist["probs"].probs))
        val = self.normalize_tensor(val, prop)
        return val

    def sample_batch(self, nodesxsample):
        vals = []
        for n_nodes in nodesxsample:
            vals.append(self.sample(int(n_nodes)).unsqueeze(0))
        vals = torch.cat(vals, dim=0)
        return vals

    def _idx2value(self, idx, params, n_bins):
        prop_range = params[1] - params[0]
        left = float(idx) / n_bins * prop_range + params[0]
        right = float(idx + 1) / n_bins * prop_range + params[0]
        val = torch.rand(1) * (right - left) + left
        return val


def prepare_context(properties_list, batch, properties_norm):
    batch_size = len(batch.batch.unique())
    device = batch.z.device
    n_nodes = batch.pos.size(0)
    context_node_nf = 0
    context_list = []
    for key in properties_list:
        mean = properties_norm[key]["mean"].to(device)
        std = properties_norm[key]["mad"].to(device)
        properties = batch.y[:, PROP_TO_IDX[key]]
        properties = (properties - mean) / std
        if len(properties) == batch_size:
            # Global feature.
            reshaped = properties[batch.batch]
            if reshaped.size() == (n_nodes,):
                reshaped = reshaped.unsqueeze(1)
            context_list.append(reshaped)
            context_node_nf += 1
        elif len(properties) == n_nodes:
            # Node feature.
            if properties.size() == (n_nodes,):
                properties = properties.unsqueeze(1)
            context_key = properties

            context_list.append(context_key)
            context_node_nf += context_key.size(1)
        else:
            raise ValueError("Invalid tensor size, more than 3 dimensions.")
    # Concatenate
    context = torch.cat(context_list, dim=1)
    assert context.size(1) == context_node_nf
    return context
