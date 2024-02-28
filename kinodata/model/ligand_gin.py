from torch import Tensor
from torch.nn import Module

from torch_geometric.nn.aggr import Aggregation
from torch_geometric.data import HeteroData
from torch_geometric.nn import MLP

from kinodata.model.resolve import resolve_loss
from kinodata.model.regression import RegressionModel
from kinodata.configuration import Config
from kinodata.model.dti import Encoder


class LigandGNNBaseline(RegressionModel):
    def __init__(
        self,
        config: Config,
        encoder: Encoder,
        aggr: Aggregation,
    ) -> None:
        super().__init__(config)
        self.encoder = encoder
        self.aggr = aggr
        self.prediction_head = MLP(
            channel_list=[
                self.hparams.hidden_channels,
                self.hparams.hidden_channels,
                1,
            ],
            plain_last=True,
        )
        self.criterion = resolve_loss(self.hparams.loss_type)
        self.define_metrics()

    def forward(self, batch: HeteroData) -> Tensor:
        h, batch_index = self.encoder(batch)
        aggr = self.aggr.forward(h, index=batch_index)
        pred = self.prediction_head(aggr)
        return pred
