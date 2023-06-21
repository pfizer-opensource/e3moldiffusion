from os.path import join
from tqdm import tqdm
import torch
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities import rank_zero_warn
from experiments.qm9.dataset import QM9
from experiments.utils.utils import make_splits, MissingEnergyException
from torch_scatter import scatter


class QM9DataModule(LightningDataModule):
    def __init__(self, hparams, dataset=None):
        super(QM9DataModule, self).__init__()
        self.save_hyperparameters(hparams)
        self._mean, self._std = None, None
        self._saved_dataloaders = dict()
        self.dataset = dataset

    def setup(self, stage):
        dataset = QM9 if self.hparams["dataset"] == "qm9" else self.hparams["dataset"]
        self.dataset = dataset(
            self.hparams["dataset_root"], **self.hparams["dataset_arg"]
        )
        self.idx_train, self.idx_val, self.idx_test = make_splits(
            len(self.dataset),
            self.hparams["train_size"],
            self.hparams["val_size"],
            self.hparams["test_size"],
            self.hparams["seed"],
            join(self.hparams["log_dir"], "splits.npz"),
            self.hparams["splits"],
        )
        print(
            f"train {len(self.idx_train)}, val {len(self.idx_val)}, test {len(self.idx_test)}"
        )
        self.train_dataset = Subset(self.dataset, self.idx_train)
        self.val_dataset = Subset(self.dataset, self.idx_val)
        self.test_dataset = Subset(self.dataset, self.idx_test)

    def train_dataloader(self):
        return self._get_dataloader(self.train_dataset, "train")

    def val_dataloader(self):
        loaders = [self._get_dataloader(self.val_dataset, "val")]
        if (
            len(self.test_dataset) > 0
            and (self.trainer.current_epoch + 1) % self.hparams["test_interval"] == 0
        ):
            loaders.append(self._get_dataloader(self.test_dataset, "test"))
        return loaders

    def test_dataloader(self, store_dataloader=True):
        return self._get_dataloader(self.test_dataset, "test", store_dataloader=store_dataloader)

    @property
    def atomref(self):
        if hasattr(self.dataset, "get_atomref"):
            return self.dataset.get_atomref()
        return None

    def _get_dataloader(self, dataset, stage, store_dataloader=True):
        store_dataloader = (
            store_dataloader and self.trainer.reload_dataloaders_every_n_epochs <= 0
        )
        if stage in self._saved_dataloaders and store_dataloader:
            # storing the dataloaders like this breaks calls to trainer.reload_train_val_dataloaders
            # but makes it possible that the dataloaders are not recreated on every testing epoch
            return self._saved_dataloaders[stage]

        if stage == "train":
            batch_size = self.hparams["batch_size"]
            shuffle = True
        elif stage in ["val", "test"]:
            batch_size = self.hparams["inference_batch_size"]
            shuffle = False

        dl = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=self.hparams["num_workers"],
            pin_memory=True,
            shuffle=shuffle,
        )

        if store_dataloader:
            self._saved_dataloaders[stage] = dl
        return dl

    def compute_mean_mad(self, properties_list):
        if self.hparams["dataset"] == "qm9":
            dataloader = self.get_dataloader(self.train_dataset, "val")
            return self.compute_mean_mad_from_dataloader(dataloader, properties_list)
        elif (
            self.hparams["dataset"] == "qm9_1half"
            or self.hparams["dataset"] == "qm9_2half"
        ):
            dataloader = self.get_dataloader(self.val_dataset, "val")
            return self.compute_mean_mad_from_dataloader(dataloader, properties_list)
        else:
            raise Exception("Wrong dataset name")

    def compute_mean_mad_from_dataloader(self, dataloader, properties_list):
        property_norms = {}
        for property_key in properties_list:
            try:
                property_name = property_key + "_mm"
                values = getattr(dataloader.dataset[:], property_name)
            except:
                property_name = property_key
                idx = dataloader.dataset[:].label2idx[property_name]
                values = torch.tensor([data.y[:, idx] for data in dataloader.dataset[:]])

            mean = torch.mean(values)
            ma = torch.abs(values - mean)
            mad = torch.mean(ma)
            property_norms[property_key] = {}
            property_norms[property_key]["mean"] = mean
            property_norms[property_key]["mad"] = mad
            del values
        return property_norms

    def get_dataloader(self, dataset, stage):
        if stage == "train":
            batch_size = self.hparams["batch_size"]
            shuffle = True
        elif stage in ["val", "test"]:
            batch_size = self.hparams["inference_batch_size"]
            shuffle = False

        dl = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=self.hparams["num_workers"],
            pin_memory=True,
            shuffle=shuffle,
        )

        return dl
