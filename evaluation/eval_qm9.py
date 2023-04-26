import argparse
from evaluation.diffusion_utils import assert_mean_zero, remove_mean
import torch
from qm9.data import QM9DataModule
from qm9.train import Trainer
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


def test_eval(model_path, output_path):
    n_report_steps = 10

    device = "cuda"
    dataset_name = "QM9"

    # load model
    model = Trainer.load_from_checkpoint(model_path, strict=True).to(device)
    model = model.eval()

    # load hyperparameter
    hparams = torch.load(model_path)["hyper_parameters"]
    hparams = dotdict(hparams)

    # load data
    print(f"Loading {dataset_name} datamodule.")
    datamodule = QM9DataModule(hparams)
    datamodule.prepare_data()
    datamodule.setup("fit")

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

    print(f"Training set. Number of structures: {len(datamodule.train_dataset)}\n")
    print(f"Validation set. Number of structures: {len(datamodule.val_dataset)}\n")
    print(f"Test set. Number of structures: {len(datamodule.test_dataset)}\n")

    test_loader = datamodule.test_dataloader(store_dataloader=False)

    nll_epoch = 0
    n_samples = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            # Get data
            pos = data.pos.to(device)
            batch = data.batch.to(device)
            z = (
                F.one_hot(
                    data.z.squeeze().long(), num_classes=max(hparams["atom_types"]) + 1
                )
                .float()[:, hparams["atom_types"]]
                .to(device)
            )
            pos = remove_mean(pos, batch)
            assert_mean_zero(pos, batch)

            Q = (torch.zeros(0) if not hparams["include_charges"] else data.Q).to(
                z.device
            )
            h = {"categorical": z, "integer": Q}

            pos, h, delta_log_px = model.normalize(pos, h, batch)

            if properties_norm is not None:
                context = prepare_context(
                    hparams["properties_list"], data, properties_norm
                )
            else:
                context = None

            # compute loss
            loss, _ = model.compute_loss(
                pos,
                h,
                batch,
                data.ptr.to(device),
                data.edge_index.to(device),
                data.edge_attr.to(device),
                context=context.to(device),
                t0_always=True,
            )
            # Correct for normalization on x.
            assert loss.size() == delta_log_px.size()
            loss = loss - delta_log_px

            N = batch.bincount()
            log_pN = nodes_dist.log_prob(N).to(device)
            assert loss.size() == log_pN.size()
            loss = loss - log_pN

            loss = loss.mean(0)

            reg_term = torch.tensor([0.0]).to(loss.device)
            loss = loss + hparams["ode_regularization"] * reg_term

            # standard nll from forward KL
            nll_epoch += loss.item() * len(batch.unique())
            n_samples += len(batch.unique())
            if i % n_report_steps == 0:
                print(
                    f"\r Test NLL \t, iter: {i}/{len(test_loader)}, "
                    f"NLL: {nll_epoch/n_samples:.2f}"
                )

        model.analyze_and_save(
            n_samples=10000,
            batch_size=100,
            wandb=False,
            path=f"{output_path}",
        )

        start = time.time()
        if model.properties_norm is not None:
            model.save_and_sample_conditional(f"{output_path}/chain_conditional/")

        model.save_and_sample_chain(f"{output_path}/chain/")

        model.sample_different_sizes_and_save(f"{output_path}/different_sizes/")

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
        if model.properties_norm is not None:
            vis.visualize_chain(
                f"{output_path}/chain_conditional/",
                model.dataset_info,
                wandb=None,
                mode="conditional",
            )

    return nll_epoch / n_samples


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description='Data generation')
    parser.add_argument('--model-path', default="/home/julian/e3moldiffusion/qm9_logs_eqgat_normal_rbf/run0/epoch=490-step=200737.ckpt", type=str,
                        help='Path to trained model')
    parser.add_argument('--output-path', default="/home/cremej01/workspace/e3mol/eval_logs/qm9_test_output", type=str,
                        help='Path to test output')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    # Evaluate negative log-likelihood for the test partitions
    test_nll = test_eval(
        model_path=args.model_path,
        output_path=args.output_path,
    )
    print(f"Final test nll {test_nll}")


if __name__ == "__main__":
    main()
