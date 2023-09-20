import torch
import torch.nn.functional as F
from torch_geometric.utils import dense_to_sparse, sort_edge_index
from typing import Optional, List
from torch_scatter import scatter_mean
from torch_geometric.nn import radius_graph


def initialize_edge_attrs_reverse(
    edge_index_global, n, bonds_prior, num_bond_classes, device
):
    # edge types for FC graph
    j, i = edge_index_global
    mask = j < i
    mask_i = i[mask]
    mask_j = j[mask]
    nE = len(mask_i)
    edge_attr_triu = torch.multinomial(bonds_prior, num_samples=nE, replacement=True)

    j = torch.concat([mask_j, mask_i])
    i = torch.concat([mask_i, mask_j])
    edge_index_global = torch.stack([j, i], dim=0)
    edge_attr_global = torch.concat([edge_attr_triu, edge_attr_triu], dim=0)
    edge_index_global, edge_attr_global = sort_edge_index(
        edge_index=edge_index_global, edge_attr=edge_attr_global, sort_by_row=False
    )
    j, i = edge_index_global
    mask = j < i
    mask_i = i[mask]
    mask_j = j[mask]

    # some assert

    edge_attr_global_dense = torch.zeros(size=(n, n), device=device, dtype=torch.long)
    edge_attr_global_dense[
        edge_index_global[0], edge_index_global[1]
    ] = edge_attr_global
    assert (edge_attr_global_dense - edge_attr_global_dense.T).sum().float() == 0.0

    edge_attr_global = F.one_hot(edge_attr_global, num_bond_classes).float()

    return edge_attr_global, edge_index_global, mask, mask_i


def bond_guidance(
    pos,
    node_feats_in,
    temb,
    bond_model,
    batch,
    batch_edge_global,
    edge_attr_global,
    edge_index_local,
    edge_index_global,
):
    guidance_type = "logsum"
    guidance_scale = 1.0e-4
    with torch.enable_grad():
        node_feats_in = node_feats_in.detach()
        pos = pos.detach().requires_grad_(True)
        bond_prediction = bond_model(
            x=node_feats_in,
            t=temb,
            pos=pos,
            edge_index_local=edge_index_local,
            edge_index_global=edge_index_global,
            edge_attr_global=edge_attr_global,
            batch=batch,
            batch_edge_global=batch_edge_global,
        )
        if guidance_type == "ensemble":
            # TO-DO
            raise NotImplementedError
        elif guidance_type == "logsum":
            uncertainty = torch.sigmoid(-torch.logsumexp(bond_prediction, dim=-1))
            uncertainty = (
                0.5
                * scatter_mean(
                    uncertainty,
                    index=edge_index_global[1],
                    dim=0,
                    dim_size=pos.size(0),
                ).log()
            )
            uncertainty = scatter_mean(uncertainty, index=batch, dim=0, dim_size=bs)
            grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(uncertainty)]
            dist_shift = -torch.autograd.grad(
                [uncertainty],
                [pos],
                grad_outputs=grad_outputs,
                create_graph=False,
                retain_graph=False,
            )[0]

    return pos + guidance_scale * dist_shift


def energy_guidance(
    pos,
    node_feats_in,
    temb,
    energy_model,
    batch,
    guidance_scale=1.0e-4,
):
    with torch.enable_grad():
        node_feats_in = node_feats_in.detach()
        pos = pos.detach().requires_grad_(True)
        energy_prediction = energy_model(
            x=node_feats_in,
            t=temb,
            pos=pos,
            batch=batch,
        )["energy_pred"]

        grad_outputs: List[Optional[torch.Tensor]] = [
            torch.ones_like(energy_prediction)
        ]
        pos_shift = -torch.autograd.grad(
            [energy_prediction],
            [pos],
            grad_outputs=grad_outputs,
            create_graph=False,
            retain_graph=False,
        )[0]

    return pos + guidance_scale * pos_shift


def force_guidance(
    pos, node_feats_in, force_model, batch, guidance_scale=0.005, cutoff=7.5
):
    edge_index_local = radius_graph(
        x=pos,
        r=cutoff,
        batch=batch,
        max_num_neighbors=128,
        flow="source_to_target",
    )

    out = force_model(
        x=node_feats_in,
        pos=pos,
        batch=batch,
        edge_index=edge_index_local,
        edge_attr=None,
    )["pseudo_forces_pred"]

    pos = pos + guidance_scale * out
    return pos
