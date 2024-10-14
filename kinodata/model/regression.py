from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import torch
from torch import Tensor
import wandb
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl

from kinodata.configuration import Config
from kinodata.model.resolve import resolve_loss
from kinodata.model.resolve import resolve_optim


def cat_many(
    data: List[Dict[str, Tensor]], subset: Optional[List[str]] = None, dim: int = 0
) -> Dict[str, Tensor]:
    if subset is None:
        subset = list(data[0].keys())
    assert set(subset).issubset(data[0].keys())

    def ensure_tensor(sub_data, key):
        if isinstance(sub_data[key], torch.Tensor):
            return sub_data[key]
        if isinstance(sub_data[key], list):
            # what have i done
            return torch.tensor([int(x) for x in sub_data[key]])
        raise ValueError(sub_data, key, "cannot convert to tensor")

    return {
        key: torch.cat([ensure_tensor(sub_data, key) for sub_data in data], dim=dim)
        for key in subset
    }


class RegressionModel(pl.LightningModule):
    log_scatter_plot: bool = False
    log_test_predictions: bool = False

    def __init__(
        self,
        config: Config,
    ) -> None:
        super().__init__()
        self.config = config
        self.save_hyperparameters(config)  # triggers wandb hook
        self.define_metrics()
        self.set_criterion()

    def set_criterion(self):
        self.criterion = resolve_loss(self.config.loss_type)

    def forward(self, batch) -> Tensor:
        raise NotImplementedError

    def configure_optimizers(self):
        Opt = resolve_optim(self.hparams.optim)
        optim = Opt(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        scheduler = ReduceLROnPlateau(
            optim,
            mode="min",
            factor=self.hparams.lr_factor,
            patience=self.hparams.lr_patience,
            min_lr=self.hparams.min_lr,
        )

        return [optim], [
            {
                "scheduler": scheduler,
                "monitor": "val/mae",
                "interval": "epoch",
                "frequency": 1,
            }
        ]

    def define_metrics(self):
        wandb.init()
        wandb.define_metric("val/mae", summary="min")
        wandb.define_metric("val/corr", summary="max")

    def training_step(self, batch, *args) -> Tensor:
        pred = self.forward(batch).view(-1, 1)
        loss = self.criterion(pred, batch.y.view(-1, 1))
        self.log("train/loss", loss, batch_size=pred.size(0), on_epoch=True)
        return loss

    def validation_step(self, batch, *args, key: str = "val"):
        pred = self.forward(batch).flatten()
        val_mae = (pred - batch.y).abs().mean()
        self.log(f"{key}/mae", val_mae, batch_size=pred.size(0), on_epoch=True)
        return {
            f"{key}/mae": val_mae,
            "pred": pred,
            "target": batch.y,
            "ident": batch.ident,
        }

    def process_eval_outputs(self, outputs) -> float:
        pred = torch.cat([output["pred"] for output in outputs], 0)
        target = torch.cat([output["target"] for output in outputs], 0)
        corr = ((pred - pred.mean()) * (target - target.mean())).mean() / (
            pred.std() * target.std()
        ).cpu().item()
        mae = (pred - target).abs().mean()
        return pred, target, corr, mae

    def validation_epoch_end(self, outputs, *args, **kwargs) -> None:
        super().validation_epoch_end(outputs)
        pred, target, corr, mae = self.process_eval_outputs(outputs)
        self.log("val/corr", corr)

        if self.log_scatter_plot:
            y_min = min(pred.min().cpu().item(), target.min().cpu().item()) - 1
            y_max = max(pred.max().cpu().item(), target.max().cpu().item()) + 1
            fig, ax = plt.subplots()
            ax.scatter(target.cpu().numpy(), pred.cpu().numpy(), s=0.7)
            ax.set_xlim(y_min, y_max)
            ax.set_ylim(y_min, y_max)
            ax.set_ylabel("Pred")
            ax.set_xlabel("Target")
            ax.set_title(f"corr={corr}")
            wandb.log({"scatter_val": wandb.Image(fig)})
            plt.close(fig)

    def predict_step(self, batch, *args):
        pred = self.forward(batch).flatten()
        return {"pred": pred, "target": batch.y.flatten()}

    def test_step(self, batch, *args, **kwargs):
        info = self.validation_step(batch, key="test")
        return info

    def test_epoch_end(self, outputs, *args, **kwargs) -> None:
        pred, target, corr, mae = self.process_eval_outputs(outputs)
        self.log("test/mae", mae)
        self.log("test/corr", corr)

        if self.log_test_predictions:
            test_predictions = wandb.Artifact("test_predictions", type="predictions")
            data = cat_many(outputs, subset=["pred", "ident"])
            values = [t.detach().cpu() for t in data.values()]
            values = torch.stack(values, dim=1)
            table = wandb.Table(columns=list(data.keys()), data=values.tolist())
            test_predictions.add(table, "predictions")
            wandb.log_artifact(test_predictions)
            pass
