import torch
from typing import Dict, List, Optional
from torch import nn, Tensor
import torch.nn.functional as F
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
        return loss[~m]

    def forward(
        self,
        true_data: Dict,
        pred_data: Dict,
        batch: Tensor,
        bond_aggregation_index: Tensor,
        weights: Optional[Tensor] = None,
        batch_reduce: bool = True
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
            #regr_loss = self.loss_non_nans(regr_loss, self.regression_key)
            regr_loss *= weights
            regr_loss = torch.sum(regr_loss, dim=0)

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
            #atoms_loss = self.loss_non_nans(atoms_loss, "atoms")
            atoms_loss *= weights
            
            if batch_reduce:
                atoms_loss = torch.sum(atoms_loss, dim=0)

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
            #charges_loss = self.loss_non_nans(charges_loss, "charges")
            charges_loss *= weights
            if batch_reduce:
                charges_loss = torch.sum(charges_loss, dim=0)

            if self.param[self.modalities.index("bonds")] == "data":
                fnc = F.cross_entropy
                take_mean = False
            else:
                fnc = F.mse_loss
                take_mean = True

            bonds_loss = fnc(pred_data["bonds"], true_data["bonds"], reduction="none")
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
            #bonds_loss = self.loss_non_nans(bonds_loss, "bonds")
            bonds_loss *= weights
            if batch_reduce:
                bonds_loss = bonds_loss.sum(dim=0)

            if "ring" in self.modalities:
                ring_loss = F.cross_entropy(
                    pred_data["ring"], true_data["ring"], reduction="none"
                )
                ring_loss = scatter_mean(
                    ring_loss, index=batch, dim=0, dim_size=batch_size
                )
                ring_loss = self.loss_non_nans(ring_loss, "ring")
                ring_loss *= weights
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
                aromatic_loss = self.loss_non_nans(aromatic_loss, "aromatic")
                aromatic_loss *= weights
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
                hybridization_loss = self.loss_non_nans(
                    hybridization_loss, "hybridization"
                )
                hybridization_loss *= weights
                hybridization_loss = torch.sum(hybridization_loss, dim=0)
            else:
                hybridization_loss = None

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
            if self.param[self.modalities.index("charges")] == "data":
                fnc = F.cross_entropy
            else:
                fnc = F.mse_loss
            charges_loss = fnc(
                pred_data["charges"], true_data["charges"], reduction="mean"
            )
            if self.param[self.modalities.index("bonds")] == "data":
                fnc = F.cross_entropy
            else:
                fnc = F.mse_loss
            bonds_loss = fnc(pred_data["bonds"], true_data["bonds"], reduction="mean")

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

        loss = {
            self.regression_key: regr_loss,
            "atoms": atoms_loss,
            "charges": charges_loss,
            "bonds": bonds_loss,
            "ring": ring_loss,
            "aromatic": aromatic_loss,
            "hybridization": hybridization_loss,
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
