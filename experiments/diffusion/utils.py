import os
import sys
from typing import List, Optional

import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import RDConfig
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol
from torch_geometric.utils import remove_self_loops, sort_edge_index
from torch_scatter import scatter_mean

from experiments.utils import concat_ligand_pocket, get_edges, zero_mean

sys.path.append(os.path.join(RDConfig.RDContribDir, "IFG"))
from ifg import identify_functional_groups


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


def get_joint_edge_attrs(
    pos,
    pos_pocket,
    batch,
    batch_pocket,
    edge_attr_global_lig,
    num_bond_classes,
    device,
):
    edge_index_global = get_edges(
        batch, batch_pocket, pos, pos_pocket, cutoff_p=5, cutoff_lp=5
    )
    edge_index_global = sort_edge_index(edge_index=edge_index_global, sort_by_row=False)
    edge_index_global, _ = remove_self_loops(edge_index_global)
    edge_attr_global = torch.zeros(
        (edge_index_global.size(1), num_bond_classes),
        dtype=torch.float32,
        device=device,
    )
    edge_mask = (edge_index_global[0] < len(batch)) & (
        edge_index_global[1] < len(batch)
    )
    edge_mask_pocket = (edge_index_global[0] >= len(batch)) & (
        edge_index_global[1] >= len(batch)
    )
    edge_attr_global[edge_mask] = edge_attr_global_lig

    if num_bond_classes == 7:
        edge_mask_ligand_pocket = (edge_index_global[0] < len(batch)) & (
            edge_index_global[1] >= len(batch)
        )
        edge_mask_pocket_ligand = (edge_index_global[0] >= len(batch)) & (
            edge_index_global[1] < len(batch)
        )
        edge_attr_global[edge_mask_pocket] = (
            torch.tensor([0, 0, 0, 0, 0, 0, 1]).float().to(edge_attr_global.device)
        )
        edge_attr_global[edge_mask_ligand_pocket] = (
            torch.tensor([0, 0, 0, 0, 0, 1, 0]).float().to(edge_attr_global.device)
        )
        edge_attr_global[edge_mask_pocket_ligand] = (
            torch.tensor([0, 0, 0, 0, 0, 1, 0]).float().to(edge_attr_global.device)
        )
    else:
        edge_attr_global[edge_mask_pocket] = (
            torch.tensor([0, 0, 0, 0, 1]).float().to(edge_attr_global.device)
        )
    # edge_attr_global[edge_mask_pocket] = 0.0

    batch_full = torch.cat([batch, batch_pocket])
    batch_edge_global = batch_full[edge_index_global[0]]  #

    return (
        edge_index_global,
        edge_attr_global,
        batch_edge_global,
        edge_mask,
        edge_mask_pocket,
    )


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

    bs = len(batch.bincount())
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
    batch_size,
    signal=1.0e-3,
    guidance_scale=100,
    optimization="minimize",
):
    with torch.enable_grad():
        node_feats_in = node_feats_in.detach()
        pos = pos.detach().requires_grad_(True)
        out = energy_model(
            x=node_feats_in,
            t=temb,
            pos=pos,
            batch=batch,
        )
        if optimization == "minimize":
            sign = -1.0
        elif optimization == "maximize":
            sign = 1.0
        else:
            raise Exception("Optimization arg needs to be 'minimize' or 'maximize'!")
        energy_prediction = sign * guidance_scale * out["property_pred"]

        grad_outputs: List[Optional[torch.Tensor]] = [
            torch.ones_like(energy_prediction)
        ]
        pos_shift = torch.autograd.grad(
            [energy_prediction],
            [pos],
            grad_outputs=grad_outputs,
            create_graph=False,
            retain_graph=False,
        )[0]

        pos_shift = zero_mean(pos_shift, batch=batch, dim_size=batch_size, dim=0)

        pos = pos + signal * pos_shift
        pos = zero_mean(pos, batch=batch, dim_size=batch_size, dim=0)

    return pos.detach()


def property_classifier_guidance(
    pos,
    atom_types,
    charge_types,
    temb,
    property_model,
    batch,
    num_atom_types,
    signal=1.0e-3,
    guidance_scale=100,
    optimization="minimize",
):
    with torch.enable_grad():
        joint_tensor = (
            torch.cat([pos, atom_types, charge_types], dim=-1)
            .detach()
            .requires_grad_(True)
        )
        out = property_model(
            x=None,
            pos=None,
            joint_tensor=joint_tensor,
            t=temb,
            batch=batch,
        )
        if optimization == "minimize":
            sign = -1.0
        elif optimization == "maximize":
            sign = 1.0
        else:
            raise Exception("Optimization arg needs to be 'minimize' or 'maximize'!")

        property_prediction = sign * guidance_scale * out["property_pred"]

        grad_outputs: List[Optional[torch.Tensor]] = [
            torch.ones_like(property_prediction)
        ]
        grad_shift = torch.autograd.grad(
            [property_prediction],
            [joint_tensor],
            grad_outputs=grad_outputs,
            create_graph=False,
            retain_graph=False,
        )[0]

    pos = pos + signal * grad_shift[:, :3]
    pos.detach_()

    signal = 1.0
    atom_types = atom_types + signal * grad_shift[:, 3 : num_atom_types + 3]
    atom_types.detach_()
    charge_types = charge_types + signal * grad_shift[:, 3 + num_atom_types :]
    charge_types.detach_()

    return pos, atom_types, charge_types


def self_guidance(
    model,
    pos=None,
    pos_pocket=None,
    atom_types=None,
    atom_types_pocket=None,
    charge_types=None,
    charges_pocket=None,
    temb=None,
    edge_index_global=None,
    edge_index_global_lig=None,
    edge_attr_global=None,
    batch=None,
    batch_pocket=None,
    batch_full=None,
    batch_edge_global=None,
    batch_size=None,
    pocket_mask=None,
    edge_mask=None,
    edge_mask_pocket=None,
    ca_mask=None,
    context=None,
    num_atom_types=None,
    signal=1.0e-3,
    guidance_scale=100,
    optimization="minimize",
):
    (
        pos_joint,
        atom_types_joint,
        charge_types_joint,
        batch_full,
        pocket_mask,
    ) = concat_ligand_pocket(
        pos,
        pos_pocket,
        atom_types,
        atom_types_pocket,
        charge_types,
        charges_pocket,
        batch,
        batch_pocket,
        sorting=False,
    )
    with torch.enable_grad():
        joint_tensor = (
            torch.cat([pos_joint, atom_types_joint, charge_types_joint], dim=-1)
            .detach()
            .requires_grad_(True)
        )
        out = model(
            x=None,
            t=temb,
            pos=None,
            joint_tensor=joint_tensor,
            edge_index_local=None,
            edge_index_global=edge_index_global,
            edge_index_global_lig=edge_index_global_lig,
            edge_attr_global=edge_attr_global,
            batch=batch_full,
            batch_edge_global=batch_edge_global,
            context=context,
            pocket_mask=pocket_mask.unsqueeze(1),
            edge_mask=edge_mask,
            edge_mask_pocket=edge_mask_pocket,
            batch_lig=batch,
            ca_mask=ca_mask,
            batch_pocket=batch_pocket,
        )
        if optimization == "minimize":
            sign = -1.0
        elif optimization == "maximize":
            sign = 1.0
        else:
            raise Exception("Optimization arg needs to be 'minimize' or 'maximize'!")
        property_pred = sign * guidance_scale * out["property_pred"]

        grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(property_pred)]
        grad_shift = torch.autograd.grad(
            [property_pred],
            [joint_tensor],
            grad_outputs=grad_outputs,
            create_graph=False,
            retain_graph=False,
        )[0]

    pos = pos + signal * grad_shift[:, :3][pocket_mask.squeeze(), :]
    pos.detach_()

    atom_types = (
        atom_types
        + signal * grad_shift[:, 3 : num_atom_types + 3][pocket_mask.squeeze(), :]
    )
    atom_types.detach_()
    charge_types = (
        charge_types
        + signal * grad_shift[:, 3 + num_atom_types :][pocket_mask.squeeze(), :]
    )
    charge_types.detach_()
    # pos = zero_mean(pos, batch=batch, dim_size=batch_size, dim=0)

    return pos, atom_types, charge_types


def extract_scaffolds_(batch_data):
    def scaffold_per_mol(mol):
        for a in mol.GetAtoms():
            a.SetIntProp("org_idx", a.GetIdx())

        scaffold = GetScaffoldForMol(mol)
        scaffold_atoms = [a.GetIntProp("org_idx") for a in scaffold.GetAtoms()]
        mask = torch.zeros(mol.GetNumAtoms(), dtype=bool)
        try:
            mask[torch.tensor(scaffold_atoms)] = 1
        except Exception as e:
            print(e)
        return mask

    batch_data.scaffold_mask = torch.hstack(
        [scaffold_per_mol(mol) for mol in batch_data.mol]
    )


def extract_func_groups_(batch_data, includeHs=True):
    def func_groups_per_mol(mol, includeHs=True):
        try:
            Chem.SanitizeMol(mol)
        except:
            pass
        fgroups = identify_functional_groups(mol)
        findices = []
        for f in fgroups:
            findices.extend(list(f.atomIds))
        if includeHs:  # include neighboring H atoms in functional groups
            findices_incl_h = []
            for fi in findices:
                hidx = [
                    n.GetIdx()
                    for n in mol.GetAtomWithIdx(fi).GetNeighbors()
                    if n.GetSymbol() == "H"
                ]
                findices_incl_h.extend([fi] + hidx)
            findices = findices_incl_h
        mask = torch.zeros(mol.GetNumAtoms(), dtype=bool)
        try:
            mask[torch.tensor(findices)] = 1
        except Exception as e:
            print(e)
        return mask

    batch_data.func_group_mask = torch.hstack(
        [func_groups_per_mol(mol, includeHs) for mol in batch_data.mol]
    )
