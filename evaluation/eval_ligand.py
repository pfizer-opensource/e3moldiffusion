import argparse
from evaluation.diffusion_utils import assert_mean_zero, remove_mean
import torch
from pocket.data import LigandPocketData
from pocket.train_diff import Trainer
import torch.nn.functional as F
import time
import evaluation.diffusion_visualiser as vis
from evaluation.diffusion_distribution import prepare_context, get_distributions
from config_file import get_dataset_info


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def test_eval(model_path, output_path, device):
    n_report_steps = 10

    dataset_name = "ligand_pocket"

    # load model
    model = Trainer.load_from_checkpoint(model_path, strict=True).to(device)
    model = model.eval()

    # load hyperparameter
    hparams = torch.load(model_path, map_location="cpu")["hyper_parameters"]
    hparams = dotdict(hparams)

    # load data
    print(f"Loading {dataset_name} datamodule.")
    datamodule = LigandPocketData(hparams)
    datamodule.prepare_data()
    datamodule.setup("fit")

    properties_norm = None
    if len(hparams.properties_list) > 0:
        properties_norm = datamodule.compute_mean_mad(hparams.properties_list)

    dataset_info = get_dataset_info(hparams.dataset, hparams.remove_hs)
    dataloader = datamodule.get_dataloader(datamodule.train_dataset, "val", shuffle=False)
    nodes_dist, prop_dist = get_distributions(hparams, dataset_info, dataloader)
    if prop_dist is not None:
        prop_dist.set_normalizer(properties_norm)

    model.nodes_dist = nodes_dist
    model.prop_dist = prop_dist
    model.dataset_info = dataset_info

    print(f"Training set. Number of structures: {len(datamodule.train_dataset)}\n")
    print(f"Validation set. Number of structures: {len(datamodule.val_dataset)}\n")
    print(f"Test set. Number of structures: {len(datamodule.test_dataset)}\n")

    test_loader = datamodule.get_dataloader(datamodule.test_dataset, "test", shuffle=False)

    model.hparams["log_dir"] = output_path

    nll_epoch = 0
    noise_loss_epoch = 0
    n_samples = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            # Get data
            ligand, pocket = model.get_ligand_and_pocket(data.to(device))

            delta_log_px, error_t_lig, error_t_pocket, SNR_weight, \
            loss_0_x_ligand, loss_0_x_pocket, loss_0_h, neg_log_const_0, \
            kl_prior, log_pN, t_int, xh_lig_hat, info = \
                model.compute_loss(ligand, pocket, return_info=True)

            log_pN = model.nodes_dist.log_prob(ligand['size'])
            # VLB objective or evaluation step
            # Note: SNR_weight should be negative
            loss_t = -model.T * 0.5 * SNR_weight * (error_t_lig + error_t_pocket)
            loss_0 = loss_0_x_ligand + loss_0_x_pocket + loss_0_h
            loss_0 = loss_0 + neg_log_const_0

            nll = loss_t + loss_0 + kl_prior
            nll = nll - delta_log_px
            nll = nll - log_pN

            noise_loss = error_t_lig.mean(0)
            loss = nll.mean(0)

            # standard nll from forward KL
            nll_epoch += loss.item() * len(ligand['size'])
            noise_loss_epoch += noise_loss.item() * len(ligand['size'])

            n_samples += len(ligand['size'])
            if i % n_report_steps == 0:
                print(
                    f"\r Test NLL \t, iter: {i}/{len(test_loader)}, "
                    f"NLL: {nll_epoch/n_samples:.2f}, "
                    f"Noise loss: {noise_loss_epoch/n_samples:.2f}"
                )

        tic = time()
        model.sample_chain_and_save_given_pocket(
            n_samples=100
            )
        print(f'Chain visualization took {time() - tic:.2f} seconds')

        tic = time()
        model.sample_and_save_given_pocket(
            n_samples=100
            )
        print(f'Sample visualization took {time() - tic:.2f} seconds')

    return nll_epoch / n_samples


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description='Data generation')
    parser.add_argument('--model-path', default="/home/cremej01/workspace/e3mol/logs/eqgat_ligand_cond/run0/last.ckpt", type=str,
                        help='Path to trained model')
    parser.add_argument('--output-path', default="/home/cremej01/workspace/e3mol/eval_logs/eqgat_ligand", type=str,
                        help='Path to test output')
    parser.add_argument('--device', default="cuda", type=str,
                        help='Which device to use.')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    # Evaluate negative log-likelihood for the test partitions
    test_nll = test_eval(
        model_path=args.model_path,
        output_path=args.output_path,
        device=args.device,
    )
    print(f"Final test nll {test_nll}")


if __name__ == "__main__":
    main()
