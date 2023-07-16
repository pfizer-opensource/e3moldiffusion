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

            coords_loss = F.mse_loss(
                pred_data["coords"], true_data["coords"], reduction="none"
            ).mean(-1)
            coords_loss = scatter_mean(
                coords_loss, index=batch, dim=0, dim_size=batch_size
            )
            coords_loss = self.loss_non_nans(coords_loss, "coords")
            coords_loss *= weights
            coords_loss = torch.sum(coords_loss, dim=0)

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
            charges_loss = self.loss_non_nans(atoms_loss, "charges")
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
        else:
            coords_loss = F.mse_loss(
                pred_data["coords"], true_data["coords"], reduction="mean"
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

        loss = {
            "coords": coords_loss,
            "atoms": atoms_loss,
            "charges": charges_loss,
            "bonds": bonds_loss,
        }

        return loss

    class PredictionLoss(nn.Module):
        def __init__(self):
            super().__init__()
            num_bond_classes = 5
            self.edge_weight = torch.tensor(
                [0.1] + [1.0] * (num_bond_classes - 1), dtype=torch.float32
            )
            self.cross_entropy = torch.nn.CrossEntropyLoss(self.edge_weight)

        def forward(self, pred, target):
            return self.cross_entropy(pred, target)
