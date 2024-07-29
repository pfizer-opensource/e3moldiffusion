import argparse
import math
import os
import re
from collections import defaultdict
from copy import deepcopy
from glob import glob
from itertools import zip_longest
from os.path import dirname, exists, join
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from pytorch_lightning.utilities import rank_zero_warn
from rdkit import Chem
from torch.optim.lr_scheduler import _LRScheduler
from torch_geometric.data import Batch, Data
from torch_geometric.nn import knn_graph, radius_graph
from torch_geometric.utils import dense_to_sparse, sort_edge_index
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils.subgraph import subgraph
from torch_scatter import scatter_add, scatter_mean
from torch_sparse import coalesce
from tqdm import tqdm

from e3moldiffusion.molfeat import atom_type_config
from experiments.data.data_info import full_atom_encoder
from experiments.data.ligand.process_crossdocked import amino_acid_dict, three_to_one
from experiments.data.ligand.utils import get_space_size, sample_atom_num
from experiments.data.utils import mol_to_torch_geometric
from experiments.molecule_utils import Molecule
from experiments.sampling.inpainting import prepare_inpainting_ligand_batch
from experiments.sampling.utils import (
    calculate_hacceptors,
    calculate_hdonors,
    calculate_logp,
    calculate_molwt,
    calculate_sa,
)

# fmt: off
# Atomic masses are based on:
#
#   Meija, J., Coplen, T., Berglund, M., et al. (2016). Atomic weights of
#   the elements 2013 (IUPAC Technical Report). Pure and Applied Chemistry,
#   88(3), pp. 265-291. Retrieved 30 Nov. 2016,
#   from doi:10.1515/pac-2015-0305
#
# Standard atomic weights are taken from Table 1: "Standard atomic weights
# 2013", with the uncertainties ignored.
# For hydrogen, helium, boron, carbon, nitrogen, oxygen, magnesium, silicon,
# sulfur, chlorine, bromine and thallium, where the weights are given as a
# range the "conventional" weights are taken from Table 3 and the ranges are
# given in the comments.
# The mass of the most stable isotope (in Table 4) is used for elements
# where there the element has no stable isotopes (to avoid NaNs): Tc, Pm,
# Po, At, Rn, Fr, Ra, Ac, everything after N
atomic_masses = np.array([
    1.0, 1.008, 4.002602, 6.94, 9.0121831,
    10.81, 12.011, 14.007, 15.999, 18.998403163,
    20.1797, 22.98976928, 24.305, 26.9815385, 28.085,
    30.973761998, 32.06, 35.45, 39.948, 39.0983,
    40.078, 44.955908, 47.867, 50.9415, 51.9961,
    54.938044, 55.845, 58.933194, 58.6934, 63.546,
    65.38, 69.723, 72.63, 74.921595, 78.971,
    79.904, 83.798, 85.4678, 87.62, 88.90584,
    91.224, 92.90637, 95.95, 97.90721, 101.07,
    102.9055, 106.42, 107.8682, 112.414, 114.818,
    118.71, 121.76, 127.6, 126.90447, 131.293,
    132.90545196, 137.327, 138.90547, 140.116, 140.90766,
    144.242, 144.91276, 150.36, 151.964, 157.25,
    158.92535, 162.5, 164.93033, 167.259, 168.93422,
    173.054, 174.9668, 178.49, 180.94788, 183.84,
    186.207, 190.23, 192.217, 195.084, 196.966569,
    200.592, 204.38, 207.2, 208.9804, 208.98243,
    209.98715, 222.01758, 223.01974, 226.02541, 227.02775,
    232.0377, 231.03588, 238.02891, 237.04817, 244.06421,
    243.06138, 247.07035, 247.07031, 251.07959, 252.083,
    257.09511, 258.09843, 259.101, 262.11, 267.122,
    268.126, 271.134, 270.133, 269.1338, 278.156,
    281.165, 281.166, 285.177, 286.182, 289.19,
    289.194, 293.204, 293.208, 294.214,
])
# fmt: on


FULL_ATOM_ENCODER = {
    "H": 0,
    "B": 1,
    "C": 2,
    "N": 3,
    "O": 4,
    "F": 5,
    "Al": 6,
    "Si": 7,
    "P": 8,
    "S": 9,
    "Cl": 10,
    "As": 11,
    "Br": 12,
    "I": 13,
    "Hg": 14,
    "Bi": 15,
}


class LoadFromFile(argparse.Action):
    # parser.add_argument('--file', type=open, action=LoadFromFile)
    def __call__(self, parser, namespace, values, option_string=None):
        if values.name.endswith("yaml") or values.name.endswith("yml"):
            with values as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
            for key in config.keys():
                if key not in namespace:
                    raise ValueError(f"Unknown argument in config file: {key}")
            if (
                "load_model" in config
                and namespace.load_model is not None
                and config["load_model"] != namespace.load_model
            ):
                rank_zero_warn(
                    f"The load model argument was specified as a command line argument "
                    f"({namespace.load_model}) and in the config file ({config['load_model']}). "
                    f"Ignoring 'load_model' from the config file and loading {namespace.load_model}."
                )
                del config["load_model"]
            namespace.__dict__.update(config)
        else:
            raise ValueError("Configuration file must end with yaml or yml")


class LoadFromCheckpoint(argparse.Action):
    # parser.add_argument('--file', type=open, action=LoadFromFile)
    def __call__(self, parser, namespace, values, option_string=None):
        hparams_path = join(dirname(values), "hparams.yaml")
        if not exists(hparams_path):
            print(
                "Failed to locate the checkpoint's hparams.yaml file. Relying on command line args."
            )
            return
        with open(hparams_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        for key in config.keys():
            if key not in namespace and key != "prior_args":
                raise ValueError(f"Unknown argument in the model checkpoint: {key}")
        namespace.__dict__.update(config)
        namespace.__dict__.update(load_model=values)


def save_argparse(args, filename, exclude=None):
    import json

    if filename.endswith("yaml") or filename.endswith("yml"):
        if isinstance(exclude, str):
            exclude = [exclude]
        args = args.__dict__.copy()
        for exl in exclude:
            del args[exl]

        ds_arg = args.get("dataset_arg")
        if ds_arg is not None and isinstance(ds_arg, str):
            args["dataset_arg"] = json.loads(args["dataset_arg"])
        yaml.dump(args, open(filename, "w"))
    else:
        raise ValueError("Configuration file should end with yaml or yml")


def chunks(list, n):
    """Yield n number of sequential chunks from l."""
    d, r = divmod(len(list), n)
    for i in range(n):
        si = (d + 1) * (i if i < r else r) + d * (0 if i < r else i - r)
        yield list[si : si + (d + 1 if i < r else d)]


def number(text):
    if text is None or text == "None":
        return None

    try:
        num_int = int(text)
    except ValueError:
        num_int = None
    num_float = float(text)

    if num_int == num_float:
        return num_int
    return num_float


class MissingEnergyException(Exception):
    pass


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


class MultiTaskLoss(torch.nn.Module):
    """https://arxiv.org/abs/1705.07115"""

    def __init__(self, reduction="mean"):
        super(MultiTaskLoss, self).__init__()
        self.is_regression = torch.Tensor([True, True, True, True])
        self.n_tasks = len(self.is_regression)
        self.log_vars = torch.nn.Parameter(torch.zeros(self.n_tasks))
        self.reduction = reduction

    def forward(self, preds, targets, loss_fn):
        dtype = preds.dtype
        device = preds.device
        stds = (torch.exp(self.log_vars) ** (1 / 2)).to(device).to(dtype)
        self.is_regression = self.is_regression.to(device).to(dtype)

        coeffs = 1 / ((self.is_regression + 1) * (stds**2))

        loss_0 = loss_fn(preds[:, 0], targets[:, 0])
        loss_1 = loss_fn(preds[:, 1], targets[:, 1])
        loss_2 = loss_fn(preds[:, 2], targets[:, 2])
        loss_3 = loss_fn(preds[:, 3], targets[:, 3])
        losses = torch.stack([loss_0, loss_1, loss_2, loss_3])

        multi_task_losses = coeffs * losses + torch.log(stds)

        if self.reduction == "sum":
            multi_task_losses = multi_task_losses.sum()
        if self.reduction == "mean":
            multi_task_losses = multi_task_losses.mean()

        return multi_task_losses


"""
usage
is_regression = torch.Tensor([True, True, False]) # True: Regression/MeanSquaredErrorLoss, False: Classification/CrossEntropyLoss
multitaskloss_instance = MultiTaskLoss(is_regression)

params = list(model.parameters()) + list(multitaskloss_instance.parameters())
torch.optim.Adam(params, lr=1e-3)

model.train()
multitaskloss.train()

losses = torch.stack(loss0, loss1, loss3)
multitaskloss = multitaskloss_instance(losses)
"""

from typing import Optional

from torch import Tensor


def one_hot(
    index: Tensor,
    num_classes: Optional[int] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    r"""Taskes a one-dimensional :obj:`index` tensor and returns a one-hot
    encoded representation of it with shape :obj:`[*, num_classes]` that has
    zeros everywhere except where the index of last dimension matches the
    corresponding value of the input tensor, in which case it will be :obj:`1`.

    .. note::
        This is a more memory-efficient version of
        :meth:`torch.nn.functional.one_hot` as you can customize the output
        :obj:`dtype`.

    Args:
        index (torch.Tensor): The one-dimensional input tensor.
        num_classes (int, optional): The total number of classes. If set to
            :obj:`None`, the number of classes will be inferred as one greater
            than the largest class value in the input tensor.
            (default: :obj:`None`)
        dtype (torch.dtype, optional): The :obj:`dtype` of the output tensor.
    """
    if index.dim() != 1:
        raise ValueError("'index' tensor needs to be one-dimensional")

    if num_classes is None:
        num_classes = int(index.max()) + 1

    out = torch.zeros((index.size(0), num_classes), dtype=dtype, device=index.device)
    return out.scatter_(1, index.unsqueeze(1), 1)


def coalesce_edges(edge_index, bond_edge_index, bond_edge_attr, n):
    edge_attr = torch.full(
        size=(edge_index.size(-1),),
        fill_value=0,
        device=edge_index.device,
        dtype=torch.long,
    )
    edge_index = torch.cat([edge_index, bond_edge_index], dim=-1)
    edge_attr = torch.cat([edge_attr, bond_edge_attr], dim=0)
    edge_index, edge_attr = coalesce(
        index=edge_index, value=edge_attr, m=n, n=n, op="max"
    )
    return edge_index, edge_attr


def get_global_ligand_pocket_index(
    pos_perturbed,
    batch,
    batch_pocket,
    batch_full,
    edge_index_global_lig,
    edge_attr_global_perturbed_lig,
    cutoff=5,
    max_num_neighbors=128,
    num_bond_classes=5,
    device="cuda",
):
    # Global interaction Ligand-Pocket
    edge_index_global = (
        torch.eq(batch_full.unsqueeze(0), batch_full.unsqueeze(-1))
        .int()
        .fill_diagonal_(0)
    )
    edge_index_global, _ = dense_to_sparse(edge_index_global)
    edge_index_global = sort_edge_index(edge_index_global, sort_by_row=False)
    edge_index_global, edge_attr_global_perturbed = coalesce_edges(
        edge_index=edge_index_global,
        bond_edge_index=edge_index_global_lig,
        bond_edge_attr=edge_attr_global_perturbed_lig,
        n=batch_full.size(0),
    )
    edge_index_global, edge_attr_global_perturbed = sort_edge_index(
        edge_index=edge_index_global,
        edge_attr=edge_attr_global_perturbed,
        sort_by_row=False,
    )
    # Local interaction Ligand-Pocket
    edge_index_local = radius_graph(
        x=pos_perturbed,
        r=cutoff,
        batch=batch_full,
        max_num_neighbors=max_num_neighbors,
    )
    edge_index_global, _ = coalesce_edges(
        edge_index=edge_index_global,
        bond_edge_index=edge_index_local,
        bond_edge_attr=torch.zeros_like(edge_index_local[0]),
        n=batch_full.size(0),
    )
    edge_index_global, _ = sort_edge_index(
        edge_index=edge_index_global,
        edge_attr=edge_attr_global_perturbed,
        sort_by_row=False,
    )
    lig_n = batch.bincount()
    pocket_n = batch_pocket.bincount()
    lig_ids = torch.cat(
        [
            torch.arange(
                lig_n[:i].sum() + pocket_n[:i].sum(),
                lig_n[:i].sum() + pocket_n[:i].sum() + lig_n[i],
            )
            for i in range(len(lig_n))
        ]
    ).to(device)
    i = (
        torch.tensor([1 if i in lig_ids else 0 for i in edge_index_global[0]])
        .bool()
        .to(device)
    )
    j = (
        torch.tensor([1 if j in lig_ids else 0 for j in edge_index_global[1]])
        .bool()
        .to(device)
    )
    edge_mask = i & j

    edge_attr_global_perturbed = F.one_hot(
        edge_attr_global_perturbed, num_classes=num_bond_classes
    ).float() * (edge_mask).unsqueeze(1)

    batch_edge_global = batch_full[edge_index_global[0]]

    return batch_edge_global, edge_index_global, edge_attr_global_perturbed, edge_mask


def get_empirical_num_nodes(dataset_info):
    num_nodes_dict = dataset_info.get("n_nodes")
    max_num_nodes = max(num_nodes_dict.keys())
    empirical_distribution_num_nodes = {
        i: num_nodes_dict.get(i) for i in range(max_num_nodes)
    }
    empirical_distribution_num_nodes_tensor = {}

    for key, value in empirical_distribution_num_nodes.items():
        if value is None:
            value = 0
        empirical_distribution_num_nodes_tensor[key] = value
    empirical_distribution_num_nodes_tensor = torch.tensor(
        list(empirical_distribution_num_nodes_tensor.values())
    ).float()
    return empirical_distribution_num_nodes_tensor


def get_list_of_edge_adjs(edge_attrs_dense, batch_num_nodes):
    ptr = torch.cat(
        [
            torch.zeros(1, device=batch_num_nodes.device, dtype=torch.long),
            batch_num_nodes.cumsum(0),
        ]
    )
    edge_tensor_lists = []
    for i in range(len(ptr) - 1):
        select_slice = slice(ptr[i].item(), ptr[i + 1].item())
        e = edge_attrs_dense[select_slice, select_slice]
        edge_tensor_lists.append(e)
    return edge_tensor_lists


def get_num_atom_types_geom(dataset: str):
    assert dataset in ["qm9", "drugs"]
    return len(atom_type_config(dataset=dataset))


def zero_mean(x: Tensor, batch: Tensor, dim_size: int, dim=0):
    out = x - scatter_mean(x, index=batch, dim=dim, dim_size=dim_size)[batch]
    return out


def remove_mean_ligand(x_lig, x_pocket, batch, batch_pocket):
    # Just subtract the center of mass of the sampled part
    mean = scatter_mean(x_lig, batch, dim=0)

    x_lig = x_lig - mean[batch]
    x_pocket = x_pocket - mean[batch_pocket]
    return x_lig, x_pocket


def remove_mean_pocket(pos_lig, pos_pocket, batch, batch_pocket):
    mean = scatter_mean(pos_pocket, batch_pocket, dim=0)
    pos_lig = pos_lig - mean[batch]
    pos_pocket = pos_pocket - mean[batch_pocket]
    return pos_lig, pos_pocket


def concat_ligand_pocket(
    pos_lig,
    pos_pocket,
    x_lig,
    x_pocket,
    c_lig,
    c_pocket,
    batch_lig,
    batch_pocket,
    sorting=False,
):
    batch_ctx = torch.cat([batch_lig, batch_pocket], dim=0)

    mask_ligand = torch.cat(
        [
            torch.ones([batch_lig.size(0)], device=batch_lig.device).bool(),
            torch.zeros([batch_pocket.size(0)], device=batch_pocket.device).bool(),
        ],
        dim=0,
    )
    pos_ctx = torch.cat([pos_lig, pos_pocket], dim=0)
    x_ctx = torch.cat([x_lig, x_pocket], dim=0)
    c_ctx = torch.cat([c_lig, c_pocket], dim=0)

    if sorting:
        sort_idx = torch.sort(batch_ctx, stable=True).indices
        mask_ligand = mask_ligand[sort_idx]
        batch_ctx = batch_ctx[sort_idx]
        pos_ctx = pos_ctx[sort_idx]
        x_ctx = x_ctx[sort_idx]
        c_ctx = c_ctx[sort_idx]

    return pos_ctx, x_ctx, c_ctx, batch_ctx, mask_ligand


def concat_ligand_pocket_addfeats(
    pos_lig,
    pos_pocket,
    atom_features_ligand,
    atom_features_pocket,
    batch_lig,
    batch_pocket,
    sorting=False,
):
    batch_ctx = torch.cat([batch_lig, batch_pocket], dim=0)

    mask_ligand = torch.cat(
        [
            torch.ones([batch_lig.size(0)], device=batch_lig.device).bool(),
            torch.zeros([batch_pocket.size(0)], device=batch_pocket.device).bool(),
        ],
        dim=0,
    )
    pos_ctx = torch.cat([pos_lig, pos_pocket], dim=0)
    atom_feats = torch.cat([atom_features_ligand, atom_features_pocket], dim=0)

    if sorting:
        sort_idx = torch.sort(batch_ctx, stable=True).indices
        mask_ligand = mask_ligand[sort_idx]
        batch_ctx = batch_ctx[sort_idx]
        pos_ctx = pos_ctx[sort_idx]
        atom_feats = atom_feats[sort_idx]

    return pos_ctx, atom_feats, batch_ctx, mask_ligand

def pocket_clash_guidance(x_l, x_p, batch_l, batch_p, sigma=2.):
    with torch.enable_grad():
        x_in = x_l.detach().requires_grad_(True)
        e = torch.exp(
            -torch.sum((x_p.view(1, -1, 3) - x_in.view(-1, 1, 3) )** 2, dim=-1) / float(sigma)
            )  # (n_l, n_p)
        connectivity_mask = batch_l.view(-1, 1) == batch_p.view(1, -1)
        e = e * connectivity_mask
        clash_loss =  -sigma * torch.log(1e-3 + e.sum(dim=-1)) # (n_l,)
        clash_loss = scatter_mean(clash_loss, batch_l).sum() # (b,)->()
        grads = torch.autograd.grad(clash_loss, x_in)[0]
    return clash_loss, grads

def assert_zero_mean(x: Tensor, batch: Tensor, dim_size: int, dim=0, eps: float = 1e-6):
    out = scatter_mean(x, index=batch, dim=dim, dim_size=dim_size).mean()
    return abs(out) < eps

def load_model(filepath, num_atom_features, device="cpu", **kwargs):

    ckpt = torch.load(filepath, map_location="cpu")
    args = ckpt["hyper_parameters"]
    model = create_model(args, num_atom_features)

    state_dict = ckpt["state_dict"]
    state_dict = {
        re.sub(r"^model\.", "", k): v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("model")
    }
    state_dict = {
        k: v
        for k, v in state_dict.items()
        if not any(x in k for x in ["prior", "sde", "cat"])
    }
    model.load_state_dict(state_dict)
    return model.to(device)


def load_model_ligand(
    filepath,
    num_atom_features,
    num_bond_classes=5,
    device="cpu",
    hparams=None,
    strict=True,
):

    ckpt = torch.load(filepath, map_location="cpu")
    args = ckpt["hyper_parameters"]

    args["use_pos_norm"] = False
    if "store_intermediate_coords" not in args.keys():
        args["store_intermediate_coords"] = hparams.store_intermediate_coords

    # NOTE: need to actually give the hparams from the current run that should be executed.
    # Right now, some defaults are overwritten from the current run:
    if not strict:
        print(
            "Overwriting some some `args` from the checkpoint to match from `hparams`"
        )
        print(
            "This is most likely done in the fine-tuning stage since not-strict model-checkpoint-loading is set."
        )
        print(
            "Its better to take the `hparams` from the current run to pass into def create_model."
        )
        for k in [
            "ligand_pocket_hidden_distance",
            "joint_property_prediction",
            "regression_property",
        ]:
            print("setting {} to {}".format(k, hparams[k]))
            args[k] = hparams[k]

    model = create_model(args, num_atom_features, num_bond_classes)

    state_dict = ckpt["state_dict"]
    state_dict = {
        re.sub(r"^model\.", "", k): v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("model")
    }
    state_dict = {
        k: v
        for k, v in state_dict.items()
        if not any(x in k for x in ["prior", "sde", "cat", "posnorm", "se3norm"])
    }
    model.load_state_dict(state_dict, strict=strict)
    return model.to(device)


def _get_state_dict(ckpt, namestr):
    out = {
        ".".join(k.split(".")[1:]): v
        for k, v in ckpt["state_dict"].items()
        if k.split(".")[0] == namestr
    }
    return out


def load_latent_encoder(filepath, max_n_nodes, device="cpu"):

    ckpt = torch.load(filepath, map_location="cpu")
    args = ckpt["hyper_parameters"]

    (
        encoder,
        latent_lin,
        graph_pooling,
        mu_logvar_z,
        node_z,
        latentmodel,
    ) = create_encoder_model(args, max_n_nodes)

    state_dict_encoder = {
        re.sub(r"^encoder\.", "", k): v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("encoder")
    }
    state_dict_encoder = {
        k: v
        for k, v in state_dict_encoder.items()
        if not any(x in k for x in ["prior", "sde", "cat", "posnorm", "se3norm"])
    }

    encoder.load_state_dict(state_dict_encoder)
    encoder = encoder.to(device)

    rest_keys = ["latent_lin", "graph_pooling", "mu_logvar_z", "node_z", "latentmodel"]
    state_dicts = {namestr: _get_state_dict(ckpt, namestr) for namestr in rest_keys}

    latent_lin.load_state_dict(state_dicts["latent_lin"])
    graph_pooling.load_state_dict(state_dicts["graph_pooling"])
    mu_logvar_z.load_state_dict(state_dicts["mu_logvar_z"])
    try:
        node_z.load_state_dict(state_dicts["node_z"])
    except Exception as e:
        print(e)
        print("Not loading the node_z state_dict because output nodes (predicting max_n_nodes) differs.")
        print("This is expected if the model is fine-tuned on a different dataset.")
    if len(state_dicts["latentmodel"]) != 0:
        latentmodel = latentmodel.load_state_dict(state_dicts["latentmodel"])

    return encoder, latent_lin, graph_pooling, mu_logvar_z, node_z, latentmodel


def create_model(hparams, num_atom_features, num_bond_classes=5):
    from e3moldiffusion.coordsatomsbonds import DenoisingEdgeNetwork

    # backward compatability:
    if "joint_property_prediction" not in hparams.keys():
        hparams["joint_property_prediction"] = False
    if "atoms_continuous" not in hparams.keys():
        hparams["atoms_continuous"] = False
    if "bonds_continuous" not in hparams.keys():
        hparams["bonds_continuous"] = False
    if "store_intermediate_coords" not in hparams.keys():
        hparams["store_intermediate_coords"] = False
    if "ligand_pocket_distance_loss" not in hparams.keys():
        hparams["ligand_pocket_distance_loss"] = False
    if "ligand_pocket_hidden_distance" not in hparams.keys():
        hparams["ligand_pocket_hidden_distance"] = False
    if "use_out_norm" not in hparams.keys():
        hparams["use_out_norm"] = True
    if "dynamic_graph" not in hparams.keys():
        hparams["dynamic_graph"] = False
    if "knn" not in hparams.keys():
        hparams["knn"] = None
    if "hybrid_knn" not in hparams.keys():
        hparams["hybrid_knn"] = None
    if "knn_with_cutoff" not in hparams.keys():
        hparams["knn_with_cutoff"] = None
    if "use_rbfs" not in hparams.keys():
        hparams["use_rbfs"] = None
    if "dataset_cutoff" not in hparams.keys():
        hparams["dataset_cutoff"] = 5.0
    if "mask_pocket_edges" not in hparams.keys():
        hparams["mask_pocket_edges"] = False
    if "model_edge_rbf_interaction" not in hparams.keys():
        hparams["model_edge_rbf_interaction"] = False
    if "model_global_edge" not in hparams.keys():
        hparams["model_global_edge"] = False
    if "use_cutoff_damping" not in hparams.keys():
        hparams["use_cutoff_damping"] = False
    if "not_strict_ckpt" not in hparams.keys():
        hparams["not_strict_ckpt"] = False

    model = DenoisingEdgeNetwork(
        hn_dim=(hparams["sdim"], hparams["vdim"]),
        num_layers=hparams["num_layers"],
        latent_dim=hparams["latent_dim"],
        use_cross_product=hparams["use_cross_product"],
        num_atom_features=num_atom_features,
        num_bond_types=num_bond_classes,
        edge_dim=hparams["edim"],
        cutoff_local=hparams["cutoff_local"],
        vector_aggr=hparams["vector_aggr"],
        fully_connected=hparams["fully_connected"],
        local_global_model=hparams["local_global_model"],
        recompute_edge_attributes=True,
        recompute_radius_graph=False,
        edge_mp=hparams["edge_mp"],
        context_mapping=hparams["context_mapping"],
        num_context_features=hparams["num_context_features"],
        coords_param=hparams["continuous_param"],
        use_pos_norm=hparams["use_pos_norm"],
        use_out_norm=hparams["use_out_norm"],
        ligand_pocket_interaction=hparams["ligand_pocket_interaction"],
        store_intermediate_coords=hparams["store_intermediate_coords"],
        distance_ligand_pocket=hparams["ligand_pocket_hidden_distance"],
        bond_prediction=hparams["bond_prediction"],
        property_prediction=hparams["property_prediction"],
        joint_property_prediction=hparams["joint_property_prediction"],
        regression_property=hparams["regression_property"],
        dynamic_graph=hparams["dynamic_graph"],
        knn=hparams["knn"],
        hybrid_knn=hparams["hybrid_knn"],
        use_rbfs=hparams["use_rbfs"],
        mask_pocket_edges=hparams["mask_pocket_edges"],
        model_edge_rbf_interaction=hparams["model_edge_rbf_interaction"],
        model_global_edge=hparams["model_global_edge"],
        use_cutoff_damping=hparams["use_cutoff_damping"],
    )
    return model


def create_encoder_model(hparams, max_n_nodes):
    from e3moldiffusion.coordsatomsbonds import (
        LatentEncoderNetwork,
        SoftMaxAttentionAggregation,
    )
    from e3moldiffusion.latent import get_latent_model
    from e3moldiffusion.modules import DenseLayer, GatedEquivBlock

    encoder = LatentEncoderNetwork(
        num_atom_features=hparams["num_atom_types"],
        num_bond_types=hparams["num_bond_classes"],
        edge_dim=hparams["edim_latent"],
        cutoff_local=hparams["cutoff_local"],
        hn_dim=(hparams["sdim_latent"], hparams["vdim_latent"]),
        num_layers=hparams["num_layers_latent"],
        vector_aggr=hparams["vector_aggr"],
        intermediate_outs=hparams["intermediate_outs"],
        use_pos_norm=hparams["use_pos_norm_latent"],
        use_out_norm=hparams["use_out_norm_latent"],
    )

    latent_lin = GatedEquivBlock(
        in_dims=(hparams["sdim_latent"], hparams["vdim_latent"]),
        out_dims=(hparams["latent_dim"], None),
    )
    m = 2 if hparams["latentmodel"] == "vae" else 1
    graph_pooling = SoftMaxAttentionAggregation(dim=hparams["latent_dim"])
    mu_logvar_z = DenseLayer(hparams["latent_dim"], m * hparams["latent_dim"])
    node_z = DenseLayer(hparams["latent_dim"], max_n_nodes)
    latentmodel = get_latent_model(hparams)

    return (encoder, latent_lin, graph_pooling, mu_logvar_z, node_z, latentmodel)


def load_energy_model(filepath, num_atom_features, device="cpu"):

    ckpt = torch.load(filepath, map_location="cpu")
    args = ckpt["hyper_parameters"]
    model = create_energy_model(args, num_atom_features)

    state_dict = ckpt["state_dict"]
    state_dict = {
        re.sub(r"^model\.", "", k): v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("model")
    }
    state_dict = {
        k: v
        for k, v in state_dict.items()
        if not any(x in k for x in ["prior", "sde", "cat"])
    }
    model.load_state_dict(state_dict)
    return model.to(device)


def create_energy_model(hparams, num_atom_features):
    from e3moldiffusion.coordsatomsbonds import EQGATEnergyNetwork

    model = EQGATEnergyNetwork(
        hn_dim=(hparams["sdim"], hparams["vdim"]),
        num_layers=hparams["num_layers"],
        num_rbfs=hparams["rbf_dim"],
        use_cross_product=hparams["use_cross_product"],
        num_atom_features=num_atom_features,
        cutoff_local=hparams["cutoff_local"],
        vector_aggr=hparams["vector_aggr"],
    )
    return model


def load_property_model(
    filepath, num_atom_features, num_bond_classes, joint_prediction=False, device="cpu"
):

    ckpt = torch.load(filepath, map_location="cpu")
    args = ckpt["hyper_parameters"]
    model = create_property_model(
        args, num_atom_features, num_bond_classes, joint_prediction=joint_prediction
    )

    state_dict = ckpt["state_dict"]
    state_dict = {
        re.sub(r"^model\.", "", k): v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("model")
    }
    state_dict = {
        k: v
        for k, v in state_dict.items()
        if not any(x in k for x in ["prior", "sde", "cat"])
    }
    model.load_state_dict(state_dict)
    return model.to(device)


def create_property_model(
    hparams, num_atom_features, num_bond_classes, joint_prediction=False
):
    from e3moldiffusion.coordsatomsbonds import EQGATEnergyNetwork

    if joint_prediction:
        from e3moldiffusion.coordsatomsbonds import DenoisingEdgeNetwork

        # backward compatability:
        if "joint_property_prediction" not in hparams.keys():
            hparams["joint_property_prediction"] = False
        if "atoms_continuous" not in hparams.keys():
            hparams["atoms_continuous"] = False
        if "bonds_continuous" not in hparams.keys():
            hparams["bonds_continuous"] = False
        if "store_intermediate_coords" not in hparams.keys():
            hparams["store_intermediate_coords"] = False
        if "ligand_pocket_distance_loss" not in hparams.keys():
            hparams["ligand_pocket_distance_loss"] = False
        if "ligand_pocket_hidden_distance" not in hparams.keys():
            hparams["ligand_pocket_hidden_distance"] = False
        if "use_out_norm" not in hparams.keys():
            hparams["use_out_norm"] = True

        model = DenoisingEdgeNetwork(
            hn_dim=(hparams["sdim"], hparams["vdim"]),
            num_layers=hparams["num_layers"],
            latent_dim=hparams["latent_dim"],
            use_cross_product=hparams["use_cross_product"],
            num_atom_features=num_atom_features,
            num_bond_types=num_bond_classes,
            edge_dim=hparams["edim"],
            cutoff_local=hparams["cutoff_local"],
            vector_aggr=hparams["vector_aggr"],
            fully_connected=hparams["fully_connected"],
            local_global_model=hparams["local_global_model"],
            recompute_edge_attributes=True,
            recompute_radius_graph=False,
            edge_mp=hparams["edge_mp"],
            context_mapping=hparams["context_mapping"],
            num_context_features=hparams["num_context_features"],
            coords_param=hparams["continuous_param"],
            use_pos_norm=hparams["use_pos_norm"],
            use_out_norm=hparams["use_out_norm"],
            ligand_pocket_interaction=hparams["ligand_pocket_interaction"],
            store_intermediate_coords=hparams["store_intermediate_coords"],
            distance_ligand_pocket=hparams["ligand_pocket_hidden_distance"],
            bond_prediction=hparams["bond_prediction"],
            property_prediction=hparams["property_prediction"],
            joint_property_prediction=hparams["joint_property_prediction"],
            regression_property=hparams["regression_property"],
            dynamic_graph=hparams["dynamic_graph"],
            knn=hparams["knn"],
            hybrid_knn=hparams["hybrid_knn"],
            use_rbfs=hparams["use_rbfs"],
            mask_pocket_edges=hparams["mask_pocket_edges"],
            model_edge_rbf_interaction=hparams["model_edge_rbf_interaction"],
            model_global_edge=hparams["model_global_edge"],
            use_cutoff_damping=hparams["use_cutoff_damping"],
        )
        return model
    else:
        model = EQGATEnergyNetwork(
            hn_dim=(hparams["sdim"], hparams["vdim"]),
            num_layers=hparams["num_layers"],
            num_rbfs=hparams["rbf_dim"],
            use_cross_product=hparams["use_cross_product"],
            num_atom_features=num_atom_features,
            cutoff_local=hparams["cutoff_local"],
            vector_aggr=hparams["vector_aggr"],
        )
    return model


def load_bond_model(filepath, dataset_statistics, device="cpu", **kwargs):

    ckpt = torch.load(filepath, map_location="cpu")
    args = ckpt["hyper_parameters"]
    model = create_bond_model(args, dataset_statistics)

    state_dict = ckpt["state_dict"]
    state_dict = {
        re.sub(r"^model\.", "", k): v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("model")
    }
    state_dict = {
        k: v
        for k, v in state_dict.items()
        if not any(x in k for x in ["prior", "sde", "cat"])
    }
    model.load_state_dict(state_dict)
    return model.to(device)


def create_bond_model(hparams, dataset_statistics):
    from e3moldiffusion.coordsatomsbonds import EdgePredictionNetwork

    num_atom_types = dataset_statistics.input_dims.X
    num_charge_classes = dataset_statistics.input_dims.C
    num_atom_features = num_atom_types + num_charge_classes
    model = EdgePredictionNetwork(
        hn_dim=(hparams["sdim"], hparams["vdim"]),
        num_layers=hparams["num_layers"],
        latent_dim=None,
        use_cross_product=hparams["use_cross_product"],
        num_atom_features=num_atom_features,
        num_bond_types=5,
        edge_dim=hparams["edim"],
        cutoff_local=hparams["cutoff_local"],
        vector_aggr=hparams["vector_aggr"],
        fully_connected=hparams["fully_connected"],
        local_global_model=hparams["local_global_model"],
        recompute_edge_attributes=True,
        recompute_radius_graph=False,
        edge_mp=hparams["edge_mp"],
    )
    return model


def truncated_exp_distribution(theta, x):
    return theta * torch.exp(-theta * x) / (1 - math.exp(-theta))


def cdf_exp_distribution(theta, x):
    return 1 - torch.exp(-theta * x)


def cdf_truncated_exp_distribution(theta, x):
    return cdf_exp_distribution(theta, x) / cdf_exp_distribution(
        theta, torch.ones_like(x)
    )


def quantile_func_truncated_exp_distribution(theta, p):
    return -1 / theta * torch.log(1 - p * (1 - math.exp(-theta)))


def sample_from_truncated_exp(theta, num_samples, device="cpu"):
    t = torch.rand((num_samples,), device=device)
    w = quantile_func_truncated_exp_distribution(theta=theta, p=t)
    return w


def t_int_to_frac(t, tmin, tmax):
    return (t - tmin) / tmax


def t_frac_to_int(t, tmin, tmax):
    return (t * tmax + tmin).long()


def dropout_node(
    edge_index: Tensor,
    batch: Tensor,
    p: float = 0.5,
    num_nodes: Optional[int] = None,
    training: bool = True,
) -> Tuple[Tensor, Tensor, Tensor]:
    r"""Randomly drops nodes from the adjacency matrix
    :obj:`edge_index` with probability :obj:`p` using samples from
    a Bernoulli distribution.

    The method returns (1) the retained :obj:`edge_index`, (2) the edge mask
    indicating which edges were retained. (3) the node mask indicating
    which nodes were retained.

    Args:
        edge_index (LongTensor): The edge indices.
        p (float, optional): Dropout probability. (default: :obj:`0.5`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        training (bool, optional): If set to :obj:`False`, this operation is a
            no-op. (default: :obj:`True`)

    :rtype: (:class:`LongTensor`, :class:`BoolTensor`, :class:`BoolTensor`)

    Examples:

        >>> edge_index = torch.tensor([[0, 1, 1, 2, 2, 3],
        ...                            [1, 0, 2, 1, 3, 2]])
        >>> edge_index, edge_mask, node_mask = dropout_node(edge_index)
        >>> edge_index
        tensor([[0, 1],
                [1, 0]])
        >>> edge_mask
        tensor([ True,  True, False, False, False, False])
        >>> node_mask
        tensor([ True,  True, False, False])
    """
    if p < 0.0 or p > 1.0:
        raise ValueError(f"Dropout probability has to be between 0 and 1 " f"(got {p}")

    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    if not training or p == 0.0:
        node_mask = edge_index.new_ones(num_nodes, dtype=torch.bool)
        edge_mask = edge_index.new_ones(edge_index.size(1), dtype=torch.bool)
        return edge_index, edge_mask, node_mask

    prob = torch.rand(num_nodes, device=edge_index.device)
    node_mask = prob > p

    bs = len(batch.unique())
    nSelect = scatter_add(node_mask.float(), index=batch, dim=0, dim_size=bs).float()
    batch_mask = ~(nSelect == 0.0)

    deleted_graphs = torch.where(~batch_mask)[0]
    if len(deleted_graphs) > 0:
        take_again = torch.zeros_like(batch)
        for graph_idx in deleted_graphs.cpu().numpy():
            graph_ids = graph_idx == batch
            take_again += graph_ids
        node_mask = (node_mask + take_again).bool()

    nSelect = scatter_add(node_mask.float(), index=batch, dim=0, dim_size=bs).float()
    batch_mask = ~(nSelect == 0.0)

    edge_index, _, edge_mask = subgraph(
        node_mask, edge_index, num_nodes=num_nodes, return_edge_mask=True
    )

    return edge_index, edge_mask, node_mask, batch_mask


def load_force_model(filepath, num_atom_features, device="cpu"):

    ckpt = torch.load(filepath, map_location="cpu")
    args = ckpt["hyper_parameters"]
    model = create_force_model(args, num_atom_features)

    state_dict = ckpt["state_dict"]
    state_dict = {
        re.sub(r"^model\.", "", k): v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("model")
    }
    state_dict = {
        k: v
        for k, v in state_dict.items()
        if not any(x in k for x in ["prior", "sde", "cat"])
    }
    model.load_state_dict(state_dict)
    return model.to(device)


def create_force_model(hparams, num_atom_features):
    from e3moldiffusion.coordsatomsbonds import EQGATForceNetwork

    model = EQGATForceNetwork(
        hn_dim=(hparams["sdim"], hparams["vdim"]),
        num_layers=hparams["num_layers"],
        num_rbfs=hparams["rbf_dim"],
        use_cross_product=hparams["use_cross_product"],
        num_atom_features=num_atom_features,
        cutoff_local=hparams["cutoff_local"],
        vector_aggr=hparams["vector_aggr"],
    )
    return model


def effective_batch_size(
    max_size, reference_batch_size, reference_size=20, sampling=False
):
    x = reference_batch_size * (reference_size / max_size) ** 2
    return math.floor(1.8 * x) if sampling else math.floor(x)


def hybrid_edge_connection(ligand_pos, protein_pos, k, ligand_index, protein_index):
    # fully-connected for ligand atoms
    k = k if len(protein_index) >= k else len(protein_index)
    dst = torch.repeat_interleave(ligand_index, len(ligand_index))
    src = ligand_index.repeat(len(ligand_index))
    mask = dst != src
    dst, src = dst[mask], src[mask]
    ll_edge_index = torch.stack([src, dst])

    # knn for ligand-protein edges
    ligand_protein_pos_dist = torch.unsqueeze(ligand_pos, 1) - torch.unsqueeze(
        protein_pos, 0
    )
    ligand_protein_pos_dist = torch.norm(ligand_protein_pos_dist, p=2, dim=-1)
    knn_p_idx = torch.topk(ligand_protein_pos_dist, k=k, largest=False, dim=1).indices
    knn_p_idx = protein_index[knn_p_idx]
    knn_l_idx = torch.unsqueeze(ligand_index, 1)
    knn_l_idx = knn_l_idx.repeat(1, k)
    lp_edge_index = torch.stack([knn_l_idx, knn_p_idx], dim=0)
    lp_edge_index = lp_edge_index.view(2, -1)
    return ll_edge_index, lp_edge_index


def batch_hybrid_edge_connection(
    full_pos, full_batch, pocket_mask, k=32, add_p_index=False
):
    batch_size = full_batch.max().item() + 1
    (
        batch_ll_edge_index,
        batch_lp_edge_index,
        batch_p_edge_index,
    ) = ([], [], [])
    with torch.no_grad():
        for i in range(batch_size):
            ligand_index = ((full_batch == i) & (pocket_mask == 1)).nonzero()[:, 0]
            protein_index = ((full_batch == i) & (pocket_mask == 0)).nonzero()[:, 0]
            ligand_pos, protein_pos = full_pos[ligand_index], full_pos[protein_index]

            ll_edge_index, lp_edge_index = hybrid_edge_connection(
                ligand_pos, protein_pos, k, ligand_index, protein_index
            )
            batch_ll_edge_index.append(ll_edge_index)
            batch_lp_edge_index.append(lp_edge_index)
            if add_p_index:
                all_pos = torch.cat([ligand_pos, protein_pos], 0)
                p_edge_index = knn_graph(all_pos, k=k, flow="source_to_target")
                p_edge_index = p_edge_index[:, p_edge_index[1] >= len(ligand_pos)]
                p_src, p_dst = p_edge_index
                all_index = torch.cat([ligand_index, protein_index], 0)
                p_edge_index = torch.stack([all_index[p_dst], all_index[p_src]], 0)
                batch_p_edge_index.append(p_edge_index)

    if add_p_index:
        edge_index = [
            torch.cat([ll, lp, p], -1)
            for ll, lp, p in zip(
                batch_ll_edge_index,
                batch_lp_edge_index,
                batch_p_edge_index,
            )
        ]
    else:
        edge_index = [
            torch.cat([ll, pl], -1)
            for ll, pl in zip(batch_ll_edge_index, batch_lp_edge_index)
        ]
    edge_index = torch.cat(edge_index, -1)
    return edge_index


def get_kNN_edges(A, top_k: int = 32):
    A[(A == 0.0)] = 1e10
    values, indices = A.topk(k=top_k, dim=-1, largest=False, sorted=True)
    AA = torch.ones_like(A)
    AA = AA * 1000.0
    AA[np.arange(AA.shape[0])[:, None], indices] = values
    return AA


def get_edges(
    batch_mask_lig,
    batch_mask_pocket,
    pos_lig,
    pos_pocket,
    cutoff_p,
    cutoff_lp,
    return_full_adj=False,
    knn=None,
    hybrid_knn=None,
    knn_with_cutoff=None,
    pocket_mask=None,
):
    if hybrid_knn is not None:
        assert not return_full_adj
        full_pos = torch.cat([pos_lig, pos_pocket], dim=0)
        full_batch = torch.cat([batch_mask_lig, batch_mask_pocket], dim=0)
        edges = batch_hybrid_edge_connection(
            full_pos, full_batch, pocket_mask, k=hybrid_knn, add_p_index=True
        )
        return edges
    else:
        adj_ligand = batch_mask_lig[:, None] == batch_mask_lig[None, :]
        adj_pocket = batch_mask_pocket[:, None] == batch_mask_pocket[None, :]
        adj_cross = batch_mask_lig[:, None] == batch_mask_pocket[None, :]

        with torch.no_grad():
            D_pocket = torch.cdist(pos_pocket, pos_pocket)
            D_cross = torch.cdist(pos_lig, pos_pocket)

        if knn is not None or knn_with_cutoff is not None:
            k = knn if knn is not None else knn_with_cutoff
            D_pocket = adj_pocket.float() * D_pocket
            D_cross = adj_cross.float() * D_cross
            D_pocket = get_kNN_edges(D_pocket, top_k=k)
            D_cross = get_kNN_edges(D_cross, top_k=k)
            if knn is not None:
                adj_pocket = adj_pocket & (D_pocket < 999)
                adj_cross = adj_cross & (D_cross < 999)

        if cutoff_p is not None and knn is None:
            adj_pocket = adj_pocket & (D_pocket <= cutoff_p)
        if cutoff_lp is not None and knn is None:
            adj_cross = adj_cross & (D_cross <= cutoff_lp)

        adj = torch.cat(
            (
                torch.cat((adj_ligand, adj_cross), dim=1),
                torch.cat((adj_cross.T, adj_pocket), dim=1),
            ),
            dim=0,
        )

        edges = torch.stack(torch.where(adj), dim=0)

        if not return_full_adj:
            return edges
        else:
            return edges, adj


def get_molecules(
    out,
    data_batch,
    edge_index_global_lig,
    num_atom_types,
    num_charge_classes,
    dataset_info,
    device,
    data_batch_pocket=None,
    pocket_name=None,
    relax_mol=False,
    max_relax_iter=200,
    sanitize=False,
    mol_device="cpu",
    context=None,
    check_validity=False,
    build_mol_with_addfeats=False,
    build_obabel_mol=False,
    while_train=False,
):
    if while_train:
        atoms_pred = out["atoms_pred"]
        atoms_pred, charges_pred = atoms_pred.split(
            [num_atom_types, num_charge_classes], dim=-1
        )
    else:
        atoms_pred = out["atoms_pred"]
        charges_pred = out["charges_pred"] if num_charge_classes > 0 else None

    batch_num_nodes = data_batch.bincount().cpu().tolist()
    pos_splits = (
        out["coords_pred"].detach().to(mol_device).split(batch_num_nodes, dim=0)
    )

    if data_batch_pocket is not None:
        batch_num_nodes_pocket = data_batch_pocket.bincount().cpu().tolist()
        pos_pocket_splits = (
            out["coords_pocket"]
            .detach()
            .to(mol_device)
            .split(batch_num_nodes_pocket, dim=0)
        )
        atom_types_integer_pocket = (
            torch.argmax(out["atoms_pocket"], dim=-1).detach().to(mol_device)
        )
        atom_types_integer_split_pocket = atom_types_integer_pocket.split(
            batch_num_nodes_pocket, dim=0
        )
    if pocket_name is None:
        pocket_name = []

    atom_types_integer = torch.argmax(atoms_pred, dim=-1).detach().to(mol_device)
    atom_types_integer_split = atom_types_integer.split(batch_num_nodes, dim=0)

    if charges_pred is not None:
        charge_types_integer = (
            torch.argmax(charges_pred, dim=-1).detach().to(mol_device)
        )
        # offset back
        charge_types_integer = charge_types_integer - dataset_info.charge_offset
        charge_types_integer_split = charge_types_integer.split(batch_num_nodes, dim=0)
    else:
        charge_types_integer_split = []

    if out["bonds_pred"] is not None:
        if out["bonds_pred"].shape[-1] > 5:
            out["bonds_pred"] = out["bonds_pred"][:, :5]
        n = data_batch.bincount().sum().item()
        edge_attrs_dense = torch.zeros(size=(n, n, 5), device=device).float()
        edge_attrs_dense[
            edge_index_global_lig[0, :], edge_index_global_lig[1, :], :
        ] = out["bonds_pred"]
        edge_attrs_dense = edge_attrs_dense.argmax(-1).detach().to(mol_device)
        edge_attrs_splits = get_list_of_edge_adjs(
            edge_attrs_dense, data_batch.bincount()
        )
    else:
        edge_attrs_splits = []

    if "aromatic_pred" in out.keys() and "hybridization_pred" in out.keys():
        add_feats = True
        aromatic_feat = out["aromatic_pred"]
        hybridization_feat = out["hybridization_pred"]
        aromatic_feat_integer = (
            torch.argmax(aromatic_feat, dim=-1).detach().to(mol_device)
        )
        aromatic_feat_integer_split = aromatic_feat_integer.split(
            batch_num_nodes, dim=0
        )

        hybridization_feat_integer = (
            torch.argmax(hybridization_feat, dim=-1).detach().to(mol_device)
        )
        hybridization_feat_integer_split = hybridization_feat_integer.split(
            batch_num_nodes, dim=0
        )
    else:
        add_feats = False

    context_split = (
        context.detach().to(mol_device).split(batch_num_nodes, dim=0)
        if context is not None
        else None
    )

    molecule_list = []

    for i, (
        positions,
        atom_types,
        charges,
        edges,
        name,
    ) in enumerate(
        zip_longest(
            pos_splits,
            atom_types_integer_split,
            charge_types_integer_split,
            edge_attrs_splits,
            pocket_name,
            fillvalue=None,
        )
    ):
        molecule = Molecule(
            atom_types=atom_types,
            positions=positions,
            charges=charges,
            bond_types=edges,
            positions_pocket=(
                pos_pocket_splits[i] if data_batch_pocket is not None else None
            ),
            atom_types_pocket=(
                atom_types_integer_split_pocket[i]
                if data_batch_pocket is not None
                else None
            ),
            pocket_name=name.split("_")[0] if name is not None else None,
            ligand_name=name.split("_")[1] if name is not None else None,
            context=context_split[i][0] if context_split is not None else None,
            is_aromatic=aromatic_feat_integer_split[i] if add_feats else None,
            hybridization=hybridization_feat_integer_split[i] if add_feats else None,
            build_mol_with_addfeats=build_mol_with_addfeats,
            dataset_info=dataset_info,
            relax_mol=relax_mol,
            max_relax_iter=max_relax_iter,
            sanitize=sanitize,
            check_validity=check_validity,
            build_obabel_mol=build_obabel_mol,
        )
        molecule_list.append(molecule)

    return molecule_list


def molecules_to_torch_geometric(
    molecule_list, add_feats, remove_hs, pocket=False, cog_proj=True
):
    data_list = []
    for molecule in molecule_list:
        data_list.append(
            mol_to_torch_geometric(
                molecule,
                full_atom_encoder,
                remove_hydrogens=remove_hs,
                cog_proj=cog_proj,
                add_pocket=pocket,
                add_ad=add_feats,
            )
        )

    # data = Batch.from_data_list(data_list)
    return data_list


def get_inp_molecules(
    inp,
    data_batch,
    edge_index_global_lig,
    dataset_info,
    device,
    mol_device="cpu",
    context=None,
):
    atoms = inp["atoms"]
    charges = inp["charges"]

    n = data_batch.bincount().sum().item()
    edge_attrs_dense = torch.zeros(size=(n, n), device=device).long()
    edge_attrs_dense[edge_index_global_lig[0, :], edge_index_global_lig[1, :]] = inp[
        "edges"
    ]
    edge_attrs_splits = get_list_of_edge_adjs(edge_attrs_dense, data_batch.bincount())

    pos_splits = (
        inp["coords"].detach().split(data_batch.bincount().cpu().tolist(), dim=0)
    )
    atom_types_integer = torch.argmax(atoms, dim=-1)
    atom_types_integer_split = atom_types_integer.detach().split(
        data_batch.bincount().cpu().tolist(), dim=0
    )
    charge_types_integer = torch.argmax(charges, dim=-1)
    # offset back
    charge_types_integer = charge_types_integer - dataset_info.charge_offset
    charge_types_integer_split = charge_types_integer.detach().split(
        data_batch.bincount().cpu().tolist(), dim=0
    )
    context_split = (
        context.split(data_batch.bincount().cpu().tolist(), dim=0)
        if context is not None
        else None
    )

    molecule_list = []
    for i, (positions, atom_types, charges, edges) in enumerate(
        zip(
            pos_splits,
            atom_types_integer_split,
            charge_types_integer_split,
            edge_attrs_splits,
        )
    ):
        molecule = Molecule(
            atom_types=atom_types.detach().to(mol_device),
            positions=positions.detach().to(mol_device),
            charges=charges.detach().to(mol_device),
            bond_types=edges.detach().to(mol_device),
            context=(
                context_split[i][0].detach().to(mol_device)
                if context_split is not None
                else None
            ),
            dataset_info=dataset_info,
        )
        molecule_list.append(molecule)

    return molecule_list


def write_sdf_file(sdf_path, molecules, extract_mol=False):
    w = Chem.SDWriter(str(sdf_path))
    for m in molecules:
        if extract_mol:
            if m.rdkit_mol is not None:
                w.write(m.rdkit_mol)
        else:
            if m is not None:
                w.write(m)
    w.close()


def unbatch_data(
    pos,
    atom_types,
    charges,
    data_batch,
    dataset_info,
):
    pos_splits = pos.detach().split(data_batch.bincount().cpu().tolist(), dim=0)

    atom_types_integer = torch.argmax(atom_types, dim=-1)
    atom_types_integer_split = atom_types_integer.detach().split(
        data_batch.bincount().cpu().tolist(), dim=0
    )
    charge_types_integer = torch.argmax(charges, dim=-1)
    # offset back
    charge_types_integer = charge_types_integer - dataset_info.charge_offset
    charge_types_integer_split = charge_types_integer.detach().split(
        data_batch.bincount().cpu().tolist(), dim=0
    )

    return pos_splits, atom_types_integer_split, charge_types_integer_split


def prepare_pocket(
    biopython_residues,
    full_atom_encoder,
    ligand_sdf=None,
    no_H=True,
    repeats=1,
    device="cuda",
):
    pocket_atoms = [
        a
        for res in biopython_residues
        for a in res.get_atoms()
        if (a.element.capitalize() in full_atom_encoder)
    ]

    pocket_coord = torch.tensor(
        np.array([a.get_coord() for a in pocket_atoms]),
        device=device,
        dtype=torch.float32,
    )
    pocket_atoms = [a.element.capitalize() for a in pocket_atoms]
    if no_H:
        indices_H = np.where(pocket_atoms == "H")
        if indices_H[0].size > 0:
            mask = np.ones(pocket_atoms.size, dtype=bool)
            mask[indices_H] = False
            pocket_atoms = pocket_atoms[mask]
            pocket_coord = pocket_coord[mask]

    pocket_types = (
        torch.tensor([full_atom_encoder[a] for a in pocket_atoms]).long().to(device)
    )
    pocket_mask = torch.repeat_interleave(
        torch.arange(repeats, device=device), len(pocket_coord)
    ).long()

    # c-alphas and residue idendity
    pocket_one_hot = []
    ca_mask = []
    for res in biopython_residues:
        for atom in res.get_atoms():
            if atom.name == "CA":
                pocket_one_hot.append(
                    np.eye(
                        1,
                        len(amino_acid_dict),
                        amino_acid_dict[three_to_one.get(res.get_resname())],
                    ).squeeze()
                )
                m = True
            else:
                m = False
            ca_mask.append(m)

    pocket_one_hot = torch.from_numpy(np.stack(pocket_one_hot, axis=0))
    ca_mask = torch.from_numpy(np.array(ca_mask, dtype=bool))
    pocket_one_hot_batch = torch.arange(repeats).repeat_interleave(
        len(pocket_one_hot), dim=0
    )
    pocket = Data(
        x_pocket=pocket_types.repeat(repeats),
        pos_pocket=pocket_coord.repeat(repeats, 1),
        pos_pocket_batch=pocket_mask,
        pocket_ca_mask=ca_mask.repeat(repeats),
        pocket_one_hot=pocket_one_hot.repeat(repeats, 1),
        pocket_one_hot_batch=pocket_one_hot_batch,
    )

    if ligand_sdf is not None:
        ligand = Chem.SDMolSupplier(str(ligand_sdf), sanitize=False)[0]
        
        if no_H:
            ligand = Chem.RemoveHs(ligand)
            Chem.Kekulize(ligand, clearAromaticFlags=True)
            
        batch = (
            torch.cat(
                [torch.tensor([i] * ligand.GetNumAtoms()) for i in range(repeats)]
            )
            .long()
            .to(device)
        )
        pocket.batch = batch

    return pocket


def get_res_coords(residues):
    # full
    full_atoms = []
    full_coords = []
    pocket_one_hot = []
    ca_mask = []
    m = False
    for res in residues:
        for atom in res.get_atoms():
            if atom.name == "CA":
                pocket_one_hot.append(
                    np.eye(
                        1,
                        len(amino_acid_dict),
                        amino_acid_dict[three_to_one.get(res.get_resname())],
                    ).squeeze()
                )
                m = True
            else:
                m = False
            ca_mask.append(m)
            full_atoms.append(atom.element)
            full_coords.append(atom.coord)
    return full_coords, full_atoms, pocket_one_hot, ca_mask


def prepare_pocket_cutoff(
    biopython_residues,
    full_atom_encoder,
    ligand_sdf=None,
    no_H=True,
    repeats=1,
    device="cuda",
    residues_10A=None,
):

    full_coords, full_atoms, pocket_one_hot, ca_mask = get_res_coords(
        biopython_residues
    )
    full_atoms = np.stack(full_atoms, axis=0)
    full_coords = np.stack(full_coords, axis=0)
    ca_mask = np.array(ca_mask, dtype=bool)

    if no_H:
        indices_H = np.where(full_atoms == "H")
        if indices_H[0].size > 0:
            mask = np.ones(full_atoms.size, dtype=bool)
            mask[indices_H] = False
            full_atoms = full_atoms[mask]
            full_coords = full_coords[mask]
            ca_mask = ca_mask[mask]
    pocket_one_hot = torch.from_numpy(np.stack(pocket_one_hot, axis=0))
    pocket_types = torch.tensor([full_atom_encoder[a] for a in full_atoms]).long()
    pocket_coord = torch.from_numpy(full_coords)
    ca_mask = torch.from_numpy(np.array(ca_mask, dtype=bool))
    pocket_one_hot_batch = torch.arange(repeats).repeat_interleave(
        len(pocket_one_hot), dim=0
    )
    pocket_mask = torch.repeat_interleave(
        torch.arange(repeats), len(pocket_coord)
    ).long()

    pocket = Data(
        x_pocket=pocket_types.repeat(repeats),
        pos_pocket=pocket_coord.repeat(repeats, 1),
        pos_pocket_batch=pocket_mask,
        pocket_ca_mask=ca_mask.repeat(repeats),
        pocket_one_hot=pocket_one_hot.repeat(repeats, 1),
        pocket_one_hot_batch=pocket_one_hot_batch,
    ).to(device)

    if residues_10A is not None:
        full_coords, full_atoms, _, _ = get_res_coords(residues_10A)
        full_coords = np.stack(full_coords, axis=0)
        full_atoms = np.stack(full_atoms, axis=0)
        if no_H:
            indices_H = np.where(full_atoms == "H")
            if indices_H[0].size > 0:
                mask = np.ones(full_atoms.size, dtype=bool)
                mask[indices_H] = False
                full_coords = full_coords[mask]
        pocket_coord = torch.from_numpy(full_coords)
        pocket_mask = torch.repeat_interleave(
            torch.arange(repeats), len(pocket_coord)
        ).long()
        pocket_coord = pocket_coord.repeat(repeats, 1)
        _num_nodes_pockets = pocket_mask.bincount()
        _pos_pocket_splits = pocket_coord.split(
            _num_nodes_pockets.cpu().numpy().tolist(), dim=0
        )
        num_nodes_lig = torch.tensor(
            [
                sample_atom_num(
                    get_space_size(n.cpu().numpy()),
                    cutoff=10,
                )
                for n in _pos_pocket_splits
            ]
        ).to(device)
        pocket.ligand_sizes = num_nodes_lig

    if ligand_sdf is not None:
        ligand = Chem.SDMolSupplier(str(ligand_sdf), sanitize=False)[0]
        
        if no_H:
            ligand = Chem.RemoveHs(ligand)
            Chem.Kekulize(ligand, clearAromaticFlags=True)
            
        batch = (
            torch.cat(
                [torch.tensor([i] * ligand.GetNumAtoms()) for i in range(repeats)]
            )
            .long()
            .to(device)
        )
        pocket.batch = batch

    return pocket


def sdfs_to_molecules(sdf_path, remove_hs=True):
    dataset_info = {"atom_decoder": FULL_ATOM_ENCODER}
    mols = []

    sdf_files = glob(os.path.join(sdf_path, "*.sdf"))

    for sdf in tqdm(sdf_files):
        ligands = Chem.SDMolSupplier(sdf)
        for mol in ligands:
            if mol is not None:
                Chem.Kekulize(mol, clearAromaticFlags=True)
                atoms = []
                charges = []
                for atom in mol.GetAtoms():
                    atoms.append(FULL_ATOM_ENCODER[atom.GetSymbol()])
                    charges.append(atom.GetFormalCharge())
                atoms = torch.Tensor(atoms).long()
                charges = torch.Tensor(charges).long()
                coords = torch.from_numpy(mol.GetConformer().GetPositions()).float()

                n = coords.size(0)
                adj = torch.from_numpy(
                    Chem.rdmolops.GetAdjacencyMatrix(mol, useBO=True)
                )
                edge_index = adj.nonzero().contiguous().T
                bond_types = adj[edge_index[0], edge_index[1]]
                bond_types[bond_types == 1.5] = 4
                if remove_hs:
                    assert max(bond_types) != 4
                edge_attr = bond_types.long()
                batch = torch.zeros(len(atoms))
                bond_edge_index, bond_edge_attr = sort_edge_index(
                    edge_index=edge_index, edge_attr=edge_attr, sort_by_row=False
                )
                edge_index_global = (
                    torch.eq(batch.unsqueeze(0), batch.unsqueeze(-1))
                    .int()
                    .fill_diagonal_(0)
                )
                edge_index_global, _ = dense_to_sparse(edge_index_global)
                edge_index_global = sort_edge_index(
                    edge_index_global, sort_by_row=False
                )
                edge_index_global, edge_attr_global = coalesce_edges(
                    edge_index=edge_index_global,
                    bond_edge_index=bond_edge_index,
                    bond_edge_attr=bond_edge_attr,
                    n=n,
                )
                edge_index_global, edge_attr_global = sort_edge_index(
                    edge_index=edge_index_global,
                    edge_attr=edge_attr_global,
                    sort_by_row=False,
                )
                edge_attrs_dense = torch.zeros(size=(n, n), device="cpu").long()
                edge_attrs_dense[edge_index_global[0, :], edge_index_global[1, :]] = (
                    edge_attr_global
                )

                molecule = Molecule(
                    atom_types=atoms,
                    positions=coords,
                    bond_types=edge_attrs_dense,
                    charges=charges,
                    rdkit_mol=mol,
                    dataset_info=dataset_info,
                )
                mols.append(molecule)
    return mols


def retrieve_interactions_per_mol(interactions_df):
    """
    Get a dictionary with interaction metrics per molecule
    """
    interactions_list = [i[-1] for i in interactions_df.columns]
    interactions_dict = defaultdict(list)
    for i in range(len(interactions_df)):
        tmp_dict = defaultdict(list)
        for k, row in enumerate(interactions_df.iloc[i, :]):
            tmp_dict[interactions_list[k]].append(row)
        for key, value in tmp_dict.items():
            interactions_dict[key].append(np.sum(value))

    interactions = {
        k: {"mean": np.mean(v), "std": np.std(v)} for k, v in interactions_dict.items()
    }
    return interactions_dict, interactions


def split_list(data, num_chunks):
    chunk_size = len(data) // num_chunks
    remainder = len(data) % num_chunks
    chunks = []
    start = 0
    for i in range(num_chunks):
        chunk_end = start + chunk_size + (1 if i < remainder else 0)
        chunks.append(data[start:chunk_end])
        start = chunk_end
    return chunks


def prepare_data(
    residues,
    sdf_file,
    dataset_info,
    hparams,
    args,
    device,
    batch_size=None,
    residues_10A=None,
):

    pocket_data = prepare_pocket_cutoff(
        residues,
        dataset_info.atom_encoder,
        no_H=hparams.remove_hs,
        repeats=batch_size,
        device=device,
        ligand_sdf=sdf_file if not args.encode_ligands else None,
        residues_10A=residues_10A,
    )
    return pocket_data


def prepare_data_and_generate_ligands(
    model,
    residues,
    sdf_file,
    dataset_info,
    hparams,
    args,
    device,
    embedding_dict=None,
    batch_size=None,
    residues_10A=None,
    inner_verbose=None,
):
    batch_size = args.batch_size if batch_size is None else batch_size
    pocket_data = prepare_data(
        residues,
        sdf_file,
        dataset_info,
        hparams,
        args,
        device,
        batch_size=batch_size,
        residues_10A=residues_10A,
    )

    if args.encode_ligands:
        assert not args.inpainting
        pocket_data, ligand_embeds = encode_ligands(
            model,
            pocket_data,
            sdf_file,
            dataset_info,
            hparams=hparams,
            args=args,
            batch_size=batch_size,
            device=device,
        )
        embedding_dict[sdf_file.stem]["seed"].append(ligand_embeds)
    
    if args.use_lipinski_context and not args.encode_ligands:
        assert not args.inpainting
        ligand_data = get_ligand_data(sdf_file, dataset_info, hparams)
        # create copies
        ligand_data = Batch.from_data_list(
            [deepcopy(ligand_data) for _ in range(batch_size)]
        )
        pocket_data.update(ligand_data)
        
    if args.inpainting:
        ligand_data = get_ligand_data(sdf_file, dataset_info, hparams, cog_proj=False)
        ligand_data = prepare_inpainting_ligand_batch(data=ligand_data,
                                                      vary_n_nodes=args.vary_n_nodes,
                                                      nodes_bias=args.n_nodes_bias,
                                                      num_graphs=batch_size,
                                                      device=device,
                                                      keep_ids=args.keep_ids
                                                      )
        pocket_data.update(ligand_data)
            
    with torch.no_grad():
        molecules = model.generate_ligands(
            pocket_data,
            num_graphs=batch_size,
            fix_n_nodes=args.fix_n_nodes,
            vary_n_nodes=args.vary_n_nodes,
            n_nodes_bias=args.n_nodes_bias,
            build_obabel_mol=args.build_obabel_mol,
            inner_verbose=inner_verbose,
            save_traj=False,
            ddpm=not args.ddim,
            eta_ddim=args.eta_ddim,
            relax_mol=args.relax_mol,
            max_relax_iter=args.max_relax_iter,
            sanitize=args.sanitize,
            ckpt_property_model=args.ckpt_property_model,
            ckpt_sa_model=args.ckpt_sa_model,
            ckpts_ensemble=args.ckpts_ensemble,
            property_classifier_guidance=args.property_classifier_guidance,
            property_classifier_guidance_complex=args.property_classifier_guidance_complex,
            property_classifier_self_guidance=args.property_classifier_self_guidance,
            classifier_guidance_scale=args.classifier_guidance_scale,
            classifier_guidance_kind=args.classifier_guidance_kind,
            classifier_guidance_period=args.classifier_guidance_period,
            sa_importance_sampling=args.sa_importance_sampling,
            sa_importance_sampling_start=args.sa_importance_sampling_start,
            sa_importance_sampling_end=args.sa_importance_sampling_end,
            sa_every_importance_t=args.sa_every_importance_t,
            sa_tau=args.sa_tau,
            property_importance_sampling=args.property_importance_sampling,
            property_importance_sampling_start=args.property_importance_sampling_start,
            property_importance_sampling_end=args.property_importance_sampling_end,
            property_every_importance_t=args.property_every_importance_t,
            property_tau=args.property_tau,
            maximize_property=not args.minimize_property,
            encode_ligand=args.encode_ligands,
            prior_n_atoms=args.prior_n_atoms,
            joint_importance_sampling=args.joint_importance_sampling,
            property_normalization=args.property_normalization,
            latent_gamma=args.latent_gamma,
            use_lipinski_context=args.use_lipinski_context,
            context_fixed=args.context_fixed,
            clash_guidance=args.clash_guidance,
            clash_guidance_start=args.clash_guidance_start,
            clash_guidance_end=args.clash_guidance_end,
            clash_guidance_scale=args.clash_guidance_scale,
            inpainting=args.inpainting,
            emd_ot=args.emd_ot,
            importance_gradient_guidance=args.importance_gradient_guidance,
        )
    del pocket_data
    torch.cuda.empty_cache()

    return molecules

def get_ligand_data(sdf_file, dataset_info, hparams, cog_proj=True):
    suppl = Chem.SDMolSupplier(str(sdf_file))
    mol = []
    for m in suppl:
        mol.append(m)
    assert len(mol) == 1
    mol = mol[0]
    ligand_data = mol_to_torch_geometric(
        mol,
        dataset_info.atom_encoder,
        smiles=Chem.MolToSmiles(mol),
        remove_hydrogens=hparams.remove_hs,
        cog_proj=cog_proj,  # only for processing the ligand-shape encode
    )
    return ligand_data
    
def encode_ligands(
    model, pocket_data, sdf_file, dataset_info, hparams, args, batch_size, device
):
    ligand_data = get_ligand_data(sdf_file, dataset_info, hparams)
    # put it into single batch
    ligand_data_ = Batch.from_data_list([ligand_data]).to(device)
    ligand_data_.mol, ligand_data_.smiles = ligand_data_.mol[0], ligand_data_.smiles[0]
    with torch.no_grad():
        ligand_embeds = model.encode_ligand(ligand_data_)
    # create copies
    ligand_data = Batch.from_data_list(
        [deepcopy(ligand_data) for _ in range(batch_size)]
    )
    pocket_data.update(ligand_data)
    return pocket_data, ligand_embeds

def get_lipinski_properties(rdmols):
    sa = torch.tensor([calculate_sa(mol) for mol in rdmols]).float()
    logp = torch.tensor([calculate_logp(mol) for mol in rdmols]).float()
    molwt = torch.tensor([calculate_molwt(mol) for mol in rdmols]).float()
    hacceptors = torch.tensor([calculate_hacceptors(mol) for mol in rdmols]).float()
    hdonors = torch.tensor([calculate_hdonors(mol) for mol in rdmols]).float()
    return torch.stack([sa, logp, molwt, hacceptors, hdonors], dim=1)


class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
    optimizer (Optimizer): Wrapped optimizer.
    first_cycle_steps (int): First cycle step size.
    cycle_mult(float): Cycle steps magnification. Default: -1.
    max_lr(float): First cycle's max learning rate. Default: 0.1.
    min_lr(float): Min learning rate. Default: 0.001.
    warmup_steps(int): Linear warmup step size. Default: 0.
    gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
    last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        first_cycle_steps: int,
        cycle_mult: float = 1.0,
        max_lr: float = 0.1,
        min_lr: float = 0.001,
        warmup_steps: int = 0,
        gamma: float = 1.0,
        last_epoch: int = -1,
    ):
        assert warmup_steps < first_cycle_steps

        self.first_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle_mult = cycle_mult  # cycle steps magnification
        self.base_max_lr = max_lr  # first max learning rate
        self.max_lr = max_lr  # max learning rate in the current cycle
        self.min_lr = min_lr  # min learning rate
        self.warmup_steps = warmup_steps  # warmup step size
        self.gamma = gamma  # decrease rate of max learning rate by cycle

        self.cur_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle = 0  # cycle count
        self.step_in_cycle = last_epoch  # step size of the current cycle

        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)

        # set learning rate min_lr
        self.init_lr()

    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.min_lr
            self.base_lrs.append(self.min_lr)

    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [
                (self.max_lr - base_lr) * self.step_in_cycle / self.warmup_steps
                + base_lr
                for base_lr in self.base_lrs
            ]
        else:
            return [
                base_lr
                + (self.max_lr - base_lr)
                * (
                    1
                    + math.cos(
                        math.pi
                        * (self.step_in_cycle - self.warmup_steps)
                        / (self.cur_cycle_steps - self.warmup_steps)
                    )
                )
                / 2
                for base_lr in self.base_lrs
            ]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = (
                    int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult)
                    + self.warmup_steps
                )
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.0:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(
                        math.log(
                            (
                                epoch / self.first_cycle_steps * (self.cycle_mult - 1)
                                + 1
                            ),
                            self.cycle_mult,
                        )
                    )
                    self.cycle = n
                    self.step_in_cycle = epoch - int(
                        self.first_cycle_steps
                        * (self.cycle_mult**n - 1)
                        / (self.cycle_mult - 1)
                    )
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (
                        n
                    )
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch

        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr
