import argparse
from evaluation.diffusion_utils import assert_mean_zero, remove_mean
import torch
from aqm.data import AQMDataModule
from aqm.train import Trainer
import torch.nn.functional as F
import time
import evaluation.diffusion_visualiser as vis
from evaluation.diffusion_distribution import get_distributions
from config_file import get_dataset_info
from rdkit import Chem
from aqm.info_data import AQMInfos
from aqm.analyze_sampling import SamplingMetrics


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def test_conditional(
    model_path, output_path, num_atoms, interpolate_prop, interpolate_range, device
):
    dataset_name = "AQM"

    # load model
    model = Trainer.load_from_checkpoint(model_path, strict=False).to(device)
    model = model.eval()

    # load hyperparameter
    hparams = torch.load(model_path)["hyper_parameters"]
    hparams = dotdict(hparams)

    # load data
    print(f"Loading {dataset_name} datamodule.")
    datamodule = AQMDataModule(hparams)
    datamodule.prepare_data()
    datamodule.setup("fit")

    dataset_statistics = AQMInfos(datamodule.dataset)

    smiles = datamodule.dataset.data.smiles
    train_idx = [int(i) for i in datamodule.idx_train]
    train_smiles = [smi for i, smi in enumerate(smiles) if i in train_idx]

    model.dataset_statistics = dataset_statistics
    model.train_smiles = train_smiles
    model.val_sampling_metrics = SamplingMetrics(
        train_smiles, dataset_statistics, test=False
    )
    properties_norm = None
    if len(hparams.properties_list) > 0:
        properties_norm = datamodule.compute_mean_mad(hparams.properties_list)

    dataset_info = get_dataset_info(hparams.dataset, hparams.remove_hs)
    dataloader = datamodule.get_dataloader(datamodule.train_dataset, "val")
    nodes_dist, prop_dist = get_distributions(hparams, dataset_info, dataloader)
    if prop_dist is not None:
        prop_dist.set_normalizer(properties_norm)

    model.nodes_dist = nodes_dist
    model.prop_dist = prop_dist
    model.dataset_info = dataset_info

    model.analyze_and_save(
        n_samples=100,
        batch_size=100,
        wandb=False,
        path=f"{output_path}",
        test_run=True,
    )

    start = time.time()

    model.save_and_sample_chain(f"{output_path}/chain/")

    model.sample_different_sizes_and_save(f"{output_path}/different_sizes/")

    if prop_dist is not None:
        model.save_and_sample_conditional(f"{output_path}/chain_conditional/")
        vis.visualize_chain(
            f"{output_path}/chain_conditional/",
            model.dataset_info,
            wandb=None,
            mode="conditional",
        )
    print(f"Sampling took {time.time() - start:.2f} seconds")

    vis.visualize(
        f"{output_path}/",
        dataset_info=model.dataset_info,
        wandb=None,
    )
    vis.visualize_chain(
        f"{output_path}/chain/",
        model.dataset_info,
        wandb=None,
    )


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description='Data generation')
    parser.add_argument('--model-path', default="/home/cremej01/workspace/e3mol/logs/eqgat_aqm_4cond_midi/run0/last.ckpt", type=str,
                        help='Path to trained model')
    parser.add_argument('--output-path', default="/home/cremej01/workspace/e3mol/eval_logs/eqgat_aqm", type=str,
                        help='Path to test output')
    parser.add_argument('--device', default="cuda", type=str,
                        help='Which device')
    parser.add_argument('--interpolate-prop', default=None, type=str, help='Which property to interpolate')
    parser.add_argument('--interpolate-range', default=None, type=list, help='Which range to interpolate a given property')
    parser.add_argument('--num-atoms', default=70, type=int, help='How many atoms per molecule for sampling')

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    # Evaluate negative log-likelihood for the test partitions
    test_conditional(
        model_path=args.model_path,
        output_path=args.output_path,
        num_atoms=args.num_atoms,
        interpolate_prop=args.interpolate_prop,
        interpolate_range=args.interpolate_range,
        device=args.device,
    )


if __name__ == "__main__":
    main()
