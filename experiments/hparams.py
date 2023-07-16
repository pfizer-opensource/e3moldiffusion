import os
from experiments.utils import LoadFromCheckpoint, LoadFromFile

DEFAULT_SAVE_DIR = os.path.join(os.getcwd(), "3DcoordsAtomsBonds_0")

if not os.path.exists(DEFAULT_SAVE_DIR):
    os.makedirs(DEFAULT_SAVE_DIR)


def add_arguments(parser):
    """Helper function to fill the parser object.

    Args:
        parser: Parser object
    Returns:
        parser: Updated parser object
    """

    # Load yaml file
    parser.add_argument(
        "--conf", "-c", type=open, action=LoadFromFile, help="Configuration yaml file"
    )  # keep first

    # Load from checkpoint
    parser.add_argument("--load-ckpt", default="", type=str)
    parser.add_argument("--load-ckpt-from-pretrained", default=None, type=str)

    # DATA and FILES
    parser.add_argument("-s", "--save-dir", default=DEFAULT_SAVE_DIR, type=str)
    parser.add_argument("--dataset", default="drugs", choices=["qm9", "drugs"])
    parser.add_argument(
        "--dataset-root", default="/hpfs/userws/cremej01/projects/data/geom"
    )
    parser.add_argument("--use-adaptive-loader", default=True, action="store_true")
    parser.add_argument("--remove-hs", default=False, action="store_true")

    # LEARNING
    parser.add_argument("-b", "--batch-size", default=32, type=int)
    parser.add_argument("-ib", "--inference-batch-size", default=32, type=int)
    parser.add_argument("--lr", default=5e-4, type=float)
    parser.add_argument("--gamma", default=0.975, type=float)
    parser.add_argument("--grad-clip-val", default=10.0, type=float)
    parser.add_argument("--frequency", default=5, type=int)
    parser.add_argument("--lr-frequency", default=5, type=int)
    parser.add_argument("--patience", default=5, type=int)
    parser.add_argument("--lr-patience", default=5, type=int)
    parser.add_argument("--cooldown", default=5, type=int)
    parser.add_argument("--lr-cooldown", default=5, type=int)
    parser.add_argument("--factor", default=0.75, type=float)
    parser.add_argument("--lr-factor", default=0.75, type=float)

    # MODEL
    parser.add_argument("--sdim", default=256, type=int)
    parser.add_argument("--vdim", default=64, type=int)
    parser.add_argument("--latent_dim", default=None, type=int)
    parser.add_argument("--rbf-dim", default=32, type=int)
    parser.add_argument("--edim", default=32, type=int)
    parser.add_argument("--vector-aggr", default="mean", type=str)
    parser.add_argument("--num-layers", default=7, type=int)
    parser.add_argument("--fully-connected", default=True, action="store_true")
    parser.add_argument("--local-global-model", default=False, action="store_true")
    parser.add_argument("--local-edge-attrs", default=False, action="store_true")
    parser.add_argument("--use-cross-product", default=False, action="store_true")
    parser.add_argument("--cutoff-local", default=7.0, type=float)
    parser.add_argument("--cutoff-global", default=10.0, type=float)

    # DIFFUSION
    parser.add_argument(
        "--continuous",
        default=False,
        action="store_true",
        help="If the diffusion process is applied on continuous time variable. Defaults to False",
    )
    parser.add_argument(
        "--schedule", default="cosine", choices=["linear", "cosine", "quad", "sigmoid"]
    )
    parser.add_argument("--eps-min", default=1e-3, type=float)
    parser.add_argument("--beta-min", default=1e-4, type=float)
    parser.add_argument("--beta-max", default=2e-2, type=float)
    parser.add_argument("--timesteps", default=500, type=int)
    parser.add_argument("--max-time", type=str, default=None)
    parser.add_argument("--masked-pretraining", default=False, action="store_true")

    # LATENT
    parser.add_argument("--vae-beta", default=1.0, type=float)
    parser.add_argument("--sdim-latent", default=256, type=int)
    parser.add_argument("--vdim-latent", default=64, type=int)
    parser.add_argument("--latent-dim", default=None, type=int)
    parser.add_argument("--edim-latent", default=32, type=int)
    parser.add_argument("--num-layers-latent", default=7, type=int)

    # GENERAL
    parser.add_argument("-i", "--id", type=int, default=0)
    parser.add_argument("-g", "--gpus", default=1, type=int)
    parser.add_argument("-e", "--num_epochs", default=300, type=int)
    parser.add_argument("--eval-freq", default=1.0, type=float)
    parser.add_argument("--test-interval", default=5, type=int)
    parser.add_argument("-nh", "--no_h", default=False, action="store_true")
    parser.add_argument("--precision", default=32, type=int)
    parser.add_argument("--detect-anomaly", default=False, action="store_true")
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument(
        "--max-num-conformers",
        default=30,
        type=int,
        help="Maximum number of conformers per molecule. \
                            Defaults to 30. Set to -1 for all conformers available in database",
    )
    parser.add_argument("--accum-batch", default=1, type=int)
    parser.add_argument("--max-num-neighbors", default=128, type=int)
    parser.add_argument("--ema-decay", default=0.9999, type=float)
    parser.add_argument("--weight-decay", default=0.9999, type=float)

    parser.add_argument("--backprop-local", default=False, action="store_true")

    return parser
