import torch
from typing import Dict, List, Optional
from torch import nn, Tensor
import torch.nn.functional as F
from torch_scatter import scatter_mean


class DiffusionLoss(nn.Module):
    def __init__(
        self, modalities: List = ["coords", "atoms", "charges", "bonds"]
    ) -> None:
        super().__init__()
        self.modalities = modalities

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
        return loss[~m]

    def forward(
        self,
        true_data: Dict,
        pred_data: Dict,
        batch: Tensor,
        bond_aggregation_index: Tensor,
        weights: Optional[Tensor] = None,
    ) -> Dict:
        batch_size = int(batch.max()) + 1

        if weights is not None:
            assert len(weights) == batch_size

            regr_loss = F.mse_loss(
                pred_data[self.regression_key],
                true_data[self.regression_key],
                reduction="none",
            ).mean(-1)
            regr_loss = scatter_mean(regr_loss, index=batch, dim=0, dim_size=batch_size)
            regr_loss = self.loss_non_nans(regr_loss, self.regression_key)
            regr_loss *= weights
            regr_loss = torch.sum(regr_loss, dim=0)

            atoms_loss = F.cross_entropy(
                pred_data["atoms"], true_data["atoms"], reduction="none"
            )
            atoms_loss = scatter_mean(
                atoms_loss, index=batch, dim=0, dim_size=batch_size
            )
            atoms_loss = self.loss_non_nans(atoms_loss, "atoms")
            atoms_loss *= weights
            atoms_loss = torch.sum(atoms_loss, dim=0)

            charges_loss = F.cross_entropy(
                pred_data["charges"], true_data["charges"], reduction="none"
            )
            charges_loss = scatter_mean(
                charges_loss, index=batch, dim=0, dim_size=batch_size
            )
            charges_loss = self.loss_non_nans(charges_loss, "charges")
            charges_loss *= weights
            charges_loss = torch.sum(charges_loss, dim=0)

            bonds_loss = F.cross_entropy(
                pred_data["bonds"], true_data["bonds"], reduction="none"
            )
            bonds_loss = 0.5 * scatter_mean(
                bonds_loss,
                index=bond_aggregation_index,
                dim=0,
                dim_size=true_data["atoms"].size(0),
            )
            bonds_loss = scatter_mean(
                bonds_loss, index=batch, dim=0, dim_size=batch_size
            )
            bonds_loss = self.loss_non_nans(bonds_loss, "bonds")
            bonds_loss *= weights
            bonds_loss = bonds_loss.sum(dim=0)
              
            # now just numHs, somehow in branch update, other features have been deleted.
            if "numHs" in self.modalities:
                numHs_loss = F.cross_entropy(pred_data["numHs"], true_data["numHs"], reduction="none")
                numHs_loss = scatter_mean(numHs_loss, index=batch, dim=0, dim_size=batch_size)
                numHs_loss = self.loss_non_nans(numHs_loss, "numHs")
                numHs_loss *= weights
                numHs_loss = torch.sum(numHs_loss, dim=0)
            else:
                numHs_loss = 0.0
                
        else:
            regr_loss = F.mse_loss(
                pred_data[self.regression_key],
                true_data[self.regression_key],
                reduction="mean",
            ).mean(-1)
            atoms_loss = F.cross_entropy(
                pred_data["atoms"], true_data["atoms"], reduction="mean"
            )
            charges_loss = F.cross_entropy(
                pred_data["charges"], true_data["charges"], reduction="mean"
            )
            bonds_loss = F.cross_entropy(
                pred_data["bonds"], true_data["bonds"], reduction="mean"
            )
            # now just numHs, somehow in branch update, other features have been deleted.
            if "numHs" in self.modalities:
                numHs_loss = F.cross_entropy(pred_data["numHs"], true_data["numHs"], reduction="mean")
            else:
                numHs_loss = 0.0

        loss = {
            self.regression_key: regr_loss,
            "atoms": atoms_loss,
            "charges": charges_loss,
            "bonds": bonds_loss,
            "numHs": numHs_loss
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
