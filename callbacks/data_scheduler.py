from copy import deepcopy

import pytorch_lightning as pl
import torch
from experiments.sampling.analyze import analyze_stability_for_molecules
from experiments.utils import (
    molecules_to_torch_geometric,
)
from torch.utils.data import Subset
from torch_geometric.data import Batch


class DataScheduler(pl.Callback):
    def __init__(self):
        self.i = 0
        self.filtered_complexes = []

    def init_train_dataset(self, trainer):
        self.orig_train_dataset = trainer.datamodule.train_dataset

    def on_train_epoch_start(self, trainer, model):
        if self.i == 0:
            self.init_train_dataset(trainer)
        self.i += 1

        def generate_ligands(data, model):
            model = deepcopy(model).to(model.device)
            model.eval()
            molecules = model.generate_ligands(
                data,
                num_graphs=model.hparams.num_ligands_per_pocket,
                fix_n_nodes=model.hparams.fix_n_nodes,
                vary_n_nodes=model.hparams.vary_n_nodes,
                n_nodes_bias=model.hparams.n_nodes_bias,
                inner_verbose=True,
                save_traj=False,
                ddpm=True,
                eta_ddim=False,
            )
            valid_molecules = analyze_stability_for_molecules(
                molecule_list=molecules,
                dataset_info=model.dataset_info,
                smiles_train=model.train_smiles,
                local_rank=0,
                return_molecules=True,
                calculate_statistics=False,
                calculate_distribution_statistics=False,
                remove_hs=model.hparams.remove_hs,
                device=model.device,
            )
            return valid_molecules

        def get_scores(molecules, model):
            model = deepcopy(model).to(model.device)
            model.eval()
            scores = model.get_scores(molecules, model)
            return scores

            # sample new ligands

        generated_complexes = []

        for data in trainer.datamodule.test_dataloader():
            data = data.to(model.device)
            data = Batch.from_data_list(
                [deepcopy(data) for _ in range(model.hparams.num_ligands_per_pocket)]
            )
            generated_complexes.extend(generate_ligands(data, model))
            break

        # scores = get_scores(generated_complexes)
        # keep_ids = [i for i in scores if i > self.hparams.docking_scores_threshold]
        # filtered_complexes = [
        #     mol for i, mol in enumerate(generated_complexes) if i in keep_ids
        # ]
        # filtered_complexes = molecules_to_torch_geometric(
        #     filtered_complexes,
        #     add_feats=self.hparams.additional_feats,
        #     remove_hs=self.hparams.remove_hs,
        #     cog_proj=False,
        # )
        # self.filtered_complexes.extend(filtered_complexes)

        filtered_complexes = molecules_to_torch_geometric(
            generated_complexes,
            add_feats=model.hparams.additional_feats,
            remove_hs=model.hparams.remove_hs,
            cog_proj=False,
        )
        self.filtered_complexes.extend(filtered_complexes)

        train_dataset = self.orig_train_dataset
        idx_train = [
            int(i)
            for i in torch.randperm(len(train_dataset))[: len(self.filtered_complexes)]
        ] + [len(train_dataset) + i for i in range(len(self.filtered_complexes))]
        train_dataset.extend(self.filtered_complexes)
        train_dataset = Subset(train_dataset, idx_train)
        trainer.datamodule.setup_(train_dataset)
