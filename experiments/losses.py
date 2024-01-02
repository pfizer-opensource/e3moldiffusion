from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_scatter import scatter_mean


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
        aux_weight: float = 0.5,
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
                pos_list = pred_data[self.regression_key]
                pos_losses = (
                    torch.nn.functional.l1_loss(
                        pos_true[None, :, :].expand(len(pos_list), -1, -1),
                        pos_list,
                        reduction="none",
                    )
                    if l1_loss
                    else torch.square(pos_true - pos_list)
                )
                pos_losses = scatter_mean(pos_losses.mean(-1), batch, -1)
                aux_loss = pos_losses[1:].mean(0)
                pos_loss = pos_losses[-1]
                regr_loss = (1 - aux_weight) * pos_loss
                regr_loss = regr_loss + aux_weight * aux_loss
                regr_loss, m = self.loss_non_nans(regr_loss, self.regression_key)
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
                donor_loss, m = self.loss_non_nans(
                    donor_loss, "donor"
                )
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
                acceptor_loss, m = self.loss_non_nans(
                    acceptor_loss, "acceptor"
                )
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
            "acceptor": acceptor_loss
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
