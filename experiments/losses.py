from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_scatter import scatter_mean, scatter_add

class DiffusionLoss(nn.Module):
    def __init__(
        self,
        modalities: List = ["coords", "atoms", "charges", "bonds"],
        param: List = ["data", "data", "data", "data"],
    ) -> None:
        super().__init__()
        assert len(modalities) == len(param)
        self.modalities = modalities
        self.param = param

        if "coords" in modalities:
            self.regression_key = "coords"
        elif "latents" in modalities:
            self.regression_key = "latents"
        else:
            raise ValueError

    def loss_non_nans(self, loss: Tensor, modality: str) -> Tensor:
        m = loss.isnan()
        if torch.any(m):
            print(f"Recovered NaNs in {modality}. Selecting NoN-Nans")
        return loss[~m], m

    def forward(
        self,
        true_data: Dict,
        pred_data: Dict,
        batch: Tensor,
        bond_aggregation_index: Tensor,
        intermediate_coords: bool = False,
        weights: Optional[Tensor] = None,
        molsize_weights: Optional[Tensor] = None,
        aux_weight: float = 1.0,
        l1_loss: bool = False,
    ) -> Dict:
        batch_size = len(batch.unique())
        charges_loss = None
        bonds_loss = None
        mulliken_loss = None
        wbo_loss = None

        if weights is not None:
            assert len(weights) == batch_size

            if intermediate_coords:
                pos_true = true_data[self.regression_key]
                pos_list = pred_data[
                    self.regression_key
                ]  # tensor of shape [num_layers, N, 3] where the last element on first axis is the final coords prediction
                pos_losses = (
                    torch.nn.functional.l1_loss(
                        pos_true[None, :, :].expand(len(pos_list), -1, -1),
                        pos_list,
                        reduction="none",
                    )
                    if l1_loss
                    else torch.square(pos_true.unsqueeze(0) - pos_list)
                )
                pos_losses = pos_losses.mean(-1)  # [num_layers, N]
                pos_losses = scatter_mean(pos_losses, batch, -1)
                # [num_layers, bs]
                aux_loss = pos_losses[:-1].mean(0)
                pos_loss = pos_losses[-1]
                regr_loss = pos_loss + aux_weight * aux_loss
                regr_loss, m = self.loss_non_nans(regr_loss, self.regression_key)
                if molsize_weights is not None:
                    regr_loss /= molsize_weights[~m]
                regr_loss *= weights[~m]
                regr_loss = torch.sum(regr_loss, dim=0)
            else:
                regr_loss = F.mse_loss(
                    pred_data[self.regression_key],
                    true_data[self.regression_key],
                    reduction="none",
                ).mean(-1)
                regr_loss = scatter_mean(
                    regr_loss, index=batch, dim=0, dim_size=batch_size
                )
                regr_loss, m = self.loss_non_nans(regr_loss, self.regression_key)

                if molsize_weights is not None:
                    regr_loss /= molsize_weights[~m]
                regr_loss *= weights[~m]
                regr_loss = torch.sum(regr_loss, dim=0)

            if "mulliken" in self.modalities:
                mulliken_loss = F.mse_loss(
                    pred_data["mulliken"],
                    true_data["mulliken"],
                    reduction="none",
                ).mean(-1)
                mulliken_loss = scatter_mean(
                    mulliken_loss, index=batch, dim=0, dim_size=batch_size
                )
                if molsize_weights is not None:
                    mulliken_loss /= molsize_weights
                mulliken_loss *= weights
                mulliken_loss = torch.sum(mulliken_loss, dim=0)

            if "wbo" in self.modalities:
                wbo_loss = F.mse_loss(
                    pred_data["wbo"],
                    true_data["wbo"],
                    reduction="none",
                ).mean(-1)
                wbo_loss = 0.5 * scatter_mean(
                    wbo_loss,
                    index=bond_aggregation_index,
                    dim=0,
                    dim_size=true_data["atoms"].size(0),
                )
                wbo_loss = scatter_mean(
                    wbo_loss, index=batch, dim=0, dim_size=batch_size
                )
                if molsize_weights is not None:
                    wbo_loss /= molsize_weights
                wbo_loss *= weights
                wbo_loss = torch.sum(wbo_loss, dim=0)

            if self.param[self.modalities.index("atoms")] == "data":
                fnc = F.cross_entropy
                take_mean = False
            else:
                fnc = F.mse_loss
                take_mean = True

            atoms_loss = fnc(pred_data["atoms"], true_data["atoms"], reduction="none")
            if take_mean:
                atoms_loss = atoms_loss.mean(dim=1)
            atoms_loss = scatter_mean(
                atoms_loss, index=batch, dim=0, dim_size=batch_size
            )
            atoms_loss, m = self.loss_non_nans(atoms_loss, "atoms")

            if molsize_weights is not None:
                atoms_loss /= molsize_weights[~m]
            atoms_loss *= weights[~m]
            atoms_loss = torch.sum(atoms_loss, dim=0)

            if "charges" in self.modalities:
                if self.param[self.modalities.index("charges")] == "data":
                    fnc = F.cross_entropy
                    take_mean = False
                else:
                    fnc = F.mse_loss
                    take_mean = True

                charges_loss = fnc(
                    pred_data["charges"], true_data["charges"], reduction="none"
                )
                if take_mean:
                    charges_loss = charges_loss.mean(dim=1)
                charges_loss = scatter_mean(
                    charges_loss, index=batch, dim=0, dim_size=batch_size
                )
                charges_loss, m = self.loss_non_nans(charges_loss, "charges")

                if molsize_weights is not None:
                    charges_loss /= molsize_weights[~m]
                charges_loss *= weights[~m]
                charges_loss = torch.sum(charges_loss, dim=0)

            if "bonds" in self.modalities:
                if self.param[self.modalities.index("bonds")] == "data":
                    fnc = F.cross_entropy
                    take_mean = False
                else:
                    fnc = F.mse_loss
                    take_mean = True

                bonds_loss = fnc(
                    pred_data["bonds"], true_data["bonds"], reduction="none"
                )
                if take_mean:
                    bonds_loss = bonds_loss.mean(dim=1)
                bonds_loss = 0.5 * scatter_mean(
                    bonds_loss,
                    index=bond_aggregation_index,
                    dim=0,
                    dim_size=true_data["atoms"].size(0),
                )
                bonds_loss = scatter_mean(
                    bonds_loss, index=batch, dim=0, dim_size=batch_size
                )
                bonds_loss, m = self.loss_non_nans(bonds_loss, "bonds")

                if molsize_weights is not None:
                    bonds_loss /= molsize_weights[~m]
                bonds_loss *= weights[~m]
                bonds_loss = bonds_loss.sum(dim=0)

            if "ring" in self.modalities:
                ring_loss = F.cross_entropy(
                    pred_data["ring"], true_data["ring"], reduction="none"
                )
                ring_loss = scatter_mean(
                    ring_loss, index=batch, dim=0, dim_size=batch_size
                )
                ring_loss, m = self.loss_non_nans(ring_loss, "ring")

                if molsize_weights is not None:
                    ring_loss /= molsize_weights[~m]
                ring_loss *= weights[~m]
                ring_loss = torch.sum(ring_loss, dim=0)
            else:
                ring_loss = None

            if "aromatic" in self.modalities:
                aromatic_loss = F.cross_entropy(
                    pred_data["aromatic"], true_data["aromatic"], reduction="none"
                )
                aromatic_loss = scatter_mean(
                    aromatic_loss, index=batch, dim=0, dim_size=batch_size
                )
                aromatic_loss, m = self.loss_non_nans(aromatic_loss, "aromatic")

                if molsize_weights is not None:
                    aromatic_loss /= molsize_weights[~m]
                aromatic_loss *= weights[~m]
                aromatic_loss = torch.sum(aromatic_loss, dim=0)
            else:
                aromatic_loss = None

            if "hybridization" in self.modalities:
                hybridization_loss = F.cross_entropy(
                    pred_data["hybridization"],
                    true_data["hybridization"],
                    reduction="none",
                )
                hybridization_loss = scatter_mean(
                    hybridization_loss, index=batch, dim=0, dim_size=batch_size
                )
                hybridization_loss, m = self.loss_non_nans(
                    hybridization_loss, "hybridization"
                )

                if molsize_weights is not None:
                    hybridization_loss /= molsize_weights[~m]
                hybridization_loss *= weights[~m]
                hybridization_loss = torch.sum(hybridization_loss, dim=0)
            else:
                hybridization_loss = None

            if "donor" in self.modalities:
                donor_loss = F.cross_entropy(
                    pred_data["donor"],
                    true_data["donor"],
                    reduction="none",
                )
                donor_loss = scatter_mean(
                    donor_loss, index=batch, dim=0, dim_size=batch_size
                )
                donor_loss, m = self.loss_non_nans(donor_loss, "donor")

                if molsize_weights is not None:
                    donor_loss /= molsize_weights[~m]
                donor_loss *= weights[~m]
                donor_loss = torch.sum(donor_loss, dim=0)
            else:
                donor_loss = None

            if "acceptor" in self.modalities:
                acceptor_loss = F.cross_entropy(
                    pred_data["acceptor"],
                    true_data["acceptor"],
                    reduction="none",
                )
                acceptor_loss = scatter_mean(
                    acceptor_loss, index=batch, dim=0, dim_size=batch_size
                )
                acceptor_loss, m = self.loss_non_nans(acceptor_loss, "acceptor")

                if molsize_weights is not None:
                    acceptor_loss /= molsize_weights[~m]
                acceptor_loss *= weights[~m]
                acceptor_loss = torch.sum(acceptor_loss, dim=0)
            else:
                acceptor_loss = None
        else:
            regr_loss = F.mse_loss(
                pred_data[self.regression_key],
                true_data[self.regression_key],
                reduction="mean",
            ).mean(-1)
            if self.param[self.modalities.index("atoms")] == "data":
                fnc = F.cross_entropy
            else:
                fnc = F.mse_loss
            atoms_loss = fnc(pred_data["atoms"], true_data["atoms"], reduction="mean")

            if "charges" in self.modalities:
                if self.param[self.modalities.index("charges")] == "data":
                    fnc = F.cross_entropy
                else:
                    fnc = F.mse_loss
                charges_loss = fnc(
                    pred_data["charges"], true_data["charges"], reduction="mean"
                )

            if "bonds" in self.modalities:
                if self.param[self.modalities.index("bonds")] == "data":
                    fnc = F.cross_entropy
                else:
                    fnc = F.mse_loss
                bonds_loss = fnc(
                    pred_data["bonds"], true_data["bonds"], reduction="mean"
                )

            if "ring" in self.modalities:
                ring_loss = F.cross_entropy(
                    pred_data["ring"], true_data["ring"], reduction="mean"
                )
            else:
                ring_loss = None

            if "aromatic" in self.modalities:
                aromatic_loss = F.cross_entropy(
                    pred_data["aromatic"], true_data["aromatic"], reduction="mean"
                )
            else:
                aromatic_loss = None

            if "hybridization" in self.modalities:
                hybridization_loss = F.cross_entropy(
                    pred_data["hybridization"],
                    true_data["hybridization"],
                    reduction="mean",
                )
            else:
                hybridization_loss = None

            if "donor" in self.modalities:
                donor_loss = F.cross_entropy(
                    pred_data["donor"],
                    true_data["donor"],
                    reduction="mean",
                )
            else:
                donor_loss = None

            if "acceptor" in self.modalities:
                acceptor_loss = F.cross_entropy(
                    pred_data["acceptor"],
                    true_data["acceptor"],
                    reduction="mean",
                )
            else:
                acceptor_loss = None

        sa_loss, prop_loss = 0.0, 0.0
        if pred_data["properties"] is not None and true_data["properties"] is not None:
            if pred_data["properties"]["sa_score"] is not None:
                sa_true, sa_pred = (
                    true_data["properties"]["sa_score"],
                    pred_data["properties"]["sa_score"],
                )
                if weights is not None:
                    sa_loss = F.mse_loss(
                        input=sa_pred.squeeze().sigmoid(),
                        target=sa_true,
                        reduction="none",
                    )
                    sa_loss = torch.mean(weights * sa_loss)
                else:
                    sa_loss = F.mse_loss(
                        input=sa_pred.squeeze().sigmoid(),
                        target=sa_true,
                    )

            if pred_data["properties"]["property"] is not None:
                prop_true, prop_pred = (
                    true_data["properties"]["property"],
                    pred_data["properties"]["property"],
                )
                if weights is not None:
                    prop_loss = F.mse_loss(
                        input=prop_pred.squeeze(dim=1),
                        target=prop_true,
                        reduction="none",
                    )
                    prop_loss = torch.mean(weights * prop_loss)
                else:
                    prop_loss = F.mse_loss(
                        input=prop_pred.squeeze(dim=1),
                        target=prop_true,
                    )

        loss = {
            self.regression_key: regr_loss,
            "atoms": atoms_loss,
            "charges": charges_loss,
            "bonds": bonds_loss,
            "ring": ring_loss,
            "aromatic": aromatic_loss,
            "hybridization": hybridization_loss,
            "mulliken": mulliken_loss,
            "wbo": wbo_loss,
            "donor": donor_loss,
            "acceptor": acceptor_loss,
            "sa": sa_loss,
            "property": prop_loss,
        }

        return loss


class EdgePredictionLoss(nn.Module):
    def __init__(
        self,
    ) -> None:
        super().__init__()

    def forward(
        self,
        true_data: Dict,
        pred_data: Dict,
    ) -> Dict:
        bonds_loss = F.cross_entropy(pred_data, true_data, reduction="mean")

        return bonds_loss
    
def LJ_potential_loss(x_l, x_p, batch_l, batch_p, N=6, eps=1e-3, clamp_max=100., w=None, full_cross=False):
    with torch.enable_grad():
        if not full_cross:
            i_l, i_p = (batch_l[:, None] == batch_p[None, :]).nonzero().T
            dm = torch.pow(x_l[i_l] - x_p[i_p], 2).sum(-1).clamp_min(eps).sqrt()
            rdm = (1 / dm.clamp_min(eps))
            vdw1 = torch.pow(rdm , 2 * N)
            vdw2 = -2 * torch.pow(rdm, N)
            energy = scatter_add(vdw1 + vdw2, i_l, dim=0) # (n_l)
        else:
            dm = torch.sqrt(torch.sum((x_p.view(1, -1, 3) - x_l.view(-1, 1, 3) )** 2, dim=-1) + eps) # (n_l, n_p)
            connectivity_mask = batch_l.view(-1, 1) == batch_p.view(1, -1)
            rdm = connectivity_mask * (1 / dm.clamp_min(eps))
            vdw1 = torch.pow(rdm, 2 * N)
            vdw2 = -2 * torch.pow(rdm, N)
            energy = torch.sum(vdw1 + vdw2, dim=-1) # (n_l)
            
        energy = torch.clamp(energy, max=clamp_max)
        energy = scatter_add(energy, batch_l, dim=0) # (b,)
        if w is None:
            w = torch.ones_like(energy)
        energy = w * energy
        grads = torch.autograd.grad(energy.sum(), dm, retain_graph=True, create_graph=True)[0]
    return energy, grads

def pocket_clash_loss(x_l, x_p, batch_l, batch_p, sigma=2., eps=1e-3, clamp_max=100., w=None, full_cross=False):
    with torch.enable_grad():
        if not full_cross:
            i_l, i_p = (batch_l[:, None] == batch_p[None, :]).nonzero().T
            dm = torch.pow(x_l[i_l] - x_p[i_p], 2).sum(-1).clamp_min(eps)
            e = torch.exp(- dm / float(sigma))
            e = scatter_add(e, i_l, dim=0) # (n_l)
        else:
            dm = torch.sum((x_p.view(1, -1, 3) - x_l.view(-1, 1, 3) )** 2, dim=-1)
            e = torch.exp(- dm / float(sigma))  # (n_l, n_p)
            connectivity_mask = batch_l.view(-1, 1) == batch_p.view(1, -1)
            e = torch.sum(e * connectivity_mask, dim=-1)
            
        energy = -sigma * torch.log(e.clamp_min(eps)) # (n_l,)
        energy = torch.clamp(energy, max=clamp_max)
        energy = scatter_mean(energy, batch_l, dim=0) # (b,)
        if w is None:
            w = torch.ones_like(energy)
        energy = w * energy
        grads = torch.autograd.grad(energy.sum(), dm, retain_graph=True, create_graph=True)[0]
    return energy, grads