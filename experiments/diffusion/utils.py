import os
import sys
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import RDConfig
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol
from torch_geometric.data import Batch
from torch_geometric.utils import remove_self_loops, sort_edge_index
from torch_scatter import scatter_mean

from experiments.data.data_info import full_atom_encoder as atom_encoder
from experiments.data.utils import mol_to_torch_geometric
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
    edge_attr_global_dense[edge_index_global[0], edge_index_global[1]] = (
        edge_attr_global
    )
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


def property_guidance_lig_pocket(
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
    pocket_mask=None,
    edge_mask=None,
    edge_mask_pocket=None,
    ca_mask=None,
    context=None,
    num_atom_types=None,
    atoms_continuous=False,
    signal=1.0e-3,
    guidance_scale=100,
    optimization="minimize",
):

    pos = pos.detach()
    pos.requires_grad = True
    pos_pocket = pos_pocket.detach()
    pos_pocket.requires_grad = False
    if atoms_continuous:
        atom_types = atom_types.detach()
        atom_types.requires_grad = True
        atom_types_pocket = atom_types_pocket.detach()
        atom_types_pocket.requires_grad = False
        charge_types = charge_types.detach()
        charge_types.requires_grad = True
        charges_pocket = charges_pocket.detach()
        charges_pocket.requires_grad = False

    with torch.enable_grad():
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
        if atoms_continuous:
            joint_tensor = torch.cat(
                [pos_joint, atom_types_joint, charge_types_joint], dim=-1
            )
            x = None
            pos_joint = None
        else:
            joint_tensor = None
            x = torch.cat([atom_types_joint, charge_types_joint], dim=-1)

        out = model(
            x=x,
            t=temb,
            pos=pos_joint,
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

        property_pred = guidance_scale * out["property_pred"]

        grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(property_pred)]
        if atoms_continuous:
            grad_shift = torch.autograd.grad(
                [property_pred],
                [joint_tensor[pocket_mask]],
                grad_outputs=grad_outputs,
                create_graph=False,
                retain_graph=False,
            )[0]
        else:
            grad_shift = torch.autograd.grad(
                [property_pred],
                [pos],
                grad_outputs=grad_outputs,
                create_graph=False,
                retain_graph=False,
            )[0]

    pos = pos + sign * signal * grad_shift[:, :3]
    pos.detach_()

    if atoms_continuous:
        atom_types = atom_types + sign * signal * grad_shift[:, 3 : num_atom_types + 3]
        atom_types.detach_()
        charge_types = (
            charge_types + sign * signal * grad_shift[:, 3 + num_atom_types :]
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


def extract_scaffolds_from_batch(batch_data):
    bs = len(batch_data.batch.bincount())
    scaffolds = []
    for mol in batch_data.mol:
        if mol.GetNumAtoms() > 1:
            scaffold = GetScaffoldForMol(mol)
            if scaffold is not None and scaffold.GetNumAtoms() > 1:
                scaffolds.append(scaffold)
            else:
                scaffolds.append(mol)
        else:
            scaffolds.append(mol)
    # scaffolds = [GetScaffoldForMol(mol) for mol in batch_data.mol]

    data_list = [
        mol_to_torch_geometric(scaffold, atom_encoder=atom_encoder)
        for scaffold in scaffolds
    ]
    batch_data = Batch.from_data_list(data_list)
    if len(batch_data.batch.bincount()) != bs:
        import pdb

        pdb.set_trace()
    return batch_data


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


def sample_prior(batch, sigma, harmonic=True):
    if harmonic:
        bid = batch["ligand"].batch
        sde = DiffusionSDE(batch.protein_sigma * sigma)

        edges = batch["ligand", "bond_edge", "ligand"].edge_index
        edges = edges[:, edges[0] < edges[1]]  # de-duplicate
        try:
            D, P = HarmonicSDE.diagonalize(
                batch["ligand"].num_nodes,
                edges=edges.T,
                lamb=sde.lamb[bid],
                ptr=batch["ligand"].ptr,
            )
        except Exception as e:
            print('batch["ligand"].num_nodes', batch["ligand"].num_nodes)
            print("batch['ligand'].size", batch["ligand"].size)
            print("batch['protein'].size", batch["protein"].batch.bincount())
            print(batch.pdb_id)
            raise e
        noise = torch.randn_like(batch["ligand"].pos)
        prior = P @ (noise / torch.sqrt(D)[:, None])
        return prior
    else:
        prior = torch.randn_like(batch["ligand"].pos)
        return prior * sigma


class DiffusionSDE:
    def __init__(self, sigma: torch.Tensor, tau_factor=5.0):
        self.lamb = 1 / sigma**2
        self.tau_factor = tau_factor

    def var(self, t):
        return (1 - torch.exp(-self.lamb * t)) / self.lamb

    def max_t(self):
        return self.tau_factor / self.lamb

    def mu_factor(self, t):
        return torch.exp(-self.lamb * t / 2)


class HarmonicSDE:
    def __init__(
        self, N=None, edges=[], antiedges=[], a=0.5, b=0.3, J=None, diagonalize=True
    ):
        self.use_cuda = False
        self.l = 1
        if not diagonalize:
            return
        if J is not None:
            J = J
            self.D, P = np.linalg.eigh(J)
            self.P = P
            self.N = self.D.size
            return

    @staticmethod
    def diagonalize(N, edges=[], antiedges=[], a=1, b=0.3, lamb=0.0, ptr=None):
        J = torch.zeros((N, N), device=edges.device)  # temporary fix
        for i, j in edges:
            J[i, i] += a
            J[j, j] += a
            J[i, j] = J[j, i] = -a
        for i, j in antiedges:
            J[i, i] -= b
            J[j, j] -= b
            J[i, j] = J[j, i] = b
        J += torch.diag(lamb)
        if ptr is None:
            return torch.linalg.eigh(J)

        Ds, Ps = [], []
        for start, end in zip(ptr[:-1], ptr[1:]):
            D, P = torch.linalg.eigh(J[start:end, start:end])
            Ds.append(D)
            Ps.append(P)
        return torch.cat(Ds), torch.block_diag(*Ps)

    def eigens(self, t):  # eigenvalues of sigma_t
        np_ = torch if self.use_cuda else np
        D = 1 / self.D * (1 - np_.exp(-t * self.D))
        t = torch.tensor(t, device="cuda").float() if self.use_cuda else t
        return np_.where(D != 0, D, t)

    def conditional(self, mask, x2):
        J_11 = self.J[~mask][:, ~mask]
        J_12 = self.J[~mask][:, mask]
        h = -J_12 @ x2
        mu = np.linalg.inv(J_11) @ h
        D, P = np.linalg.eigh(J_11)
        z = np.random.randn(*mu.shape)
        return (P / D**0.5) @ z + mu

    def A(self, t, invT=False):
        D = self.eigens(t)
        A = self.P * (D**0.5)
        if not invT:
            return A
        AinvT = self.P / (D**0.5)
        return A, AinvT

    def Sigma_inv(self, t):
        D = 1 / self.eigens(t)
        return (self.P * D) @ self.P.T

    def Sigma(self, t):
        D = self.eigens(t)
        return (self.P * D) @ self.P.T

    @property
    def J(self):
        return (self.P * self.D) @ self.P.T

    def rmsd(self, t):
        l = self.l
        D = 1 / self.D * (1 - np.exp(-t * self.D))
        return np.sqrt(3 * D[l:].mean())

    def sample(self, t, x=None, score=False, k=None, center=True, adj=False):
        l = self.l
        np_ = torch if self.use_cuda else np
        if x is None:
            if self.use_cuda:
                x = torch.zeros((self.N, 3), device="cuda").float()
            else:
                x = np.zeros((self.N, 3))
        if t == 0:
            return x
        z = (
            np.random.randn(self.N, 3)
            if not self.use_cuda
            else torch.randn(self.N, 3, device="cuda").float()
        )
        D = self.eigens(t)
        xx = self.P.T @ x
        if center:
            z[0] = 0
            xx[0] = 0
        if k:
            z[k + l :] = 0
            xx[k + l :] = 0

        out = np_.exp(-t * self.D / 2)[:, None] * xx + np_.sqrt(D)[:, None] * z

        if score:
            score = -(1 / np_.sqrt(D))[:, None] * z
            if adj:
                score = score + self.D[:, None] * out
            return self.P @ out, self.P @ score
        return self.P @ out

    def score_norm(self, t, k=None, adj=False):
        if k == 0:
            return 0
        l = self.l
        np_ = torch if self.use_cuda else np
        k = k or self.N - 1
        D = 1 / self.eigens(t)
        if adj:
            D = D * np_.exp(-self.D * t)
        return (D[l : k + l].sum() / self.N) ** 0.5

    def inject(self, t, modes):
        # Returns noise along the given modes
        z = (
            np.random.randn(self.N, 3)
            if not self.use_cuda
            else torch.randn(self.N, 3, device="cuda").float()
        )
        z[~modes] = 0
        A = self.A(t, invT=False)
        return A @ z

    def score(self, x0, xt, t):
        # Score of the diffusion kernel
        Sigma_inv = self.Sigma_inv(t)
        mu_t = (self.P * np.exp(-t * self.D / 2)) @ (self.P.T @ x0)
        return Sigma_inv @ (mu_t - xt)

    def project(self, X, k, center=False):
        l = self.l
        # Projects onto the first k nonzero modes (and optionally centers)
        D = self.P.T @ X
        D[k + l :] = 0
        if center:
            D[0] = 0
        return self.P @ D

    def unproject(self, X, mask, k, return_Pinv=False):
        # Finds the vector along the first k nonzero modes whose mask is closest to X
        l = self.l
        PP = self.P[mask, : k + l]
        Pinv = np.linalg.pinv(PP)
        out = self.P[:, : k + l] @ Pinv @ X
        if return_Pinv:
            return out, Pinv
        return out

    def energy(self, X):
        l = self.l
        return (self.D[:, None] * (self.P.T @ X) ** 2).sum(-1)[l:] / 2

    @property
    def free_energy(self):
        l = self.l
        return 3 * np.log(self.D[l:]).sum() / 2

    def KL_H(self, t):
        l = self.l
        D = self.D[l:]
        return -3 * 0.5 * (np.log(1 - np.exp(-D * t)) + np.exp(-D * t)).sum(0)
