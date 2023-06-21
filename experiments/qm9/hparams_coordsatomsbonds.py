from experiments.utils.utils import number, LoadFromFile
from experiments.utils.utils import LoadFromCheckpoint, LoadFromFile


def add_arguments(parser):
    """Helper function to fill the parser object.

    Args:
        parser: Parser object
    Returns:
        parser: Updated parser object
    """

    # Load yaml file
    parser.add_argument(
        "--load-model",
        action=LoadFromCheckpoint,
        help="Restart training using a model checkpoint",
    )  # keep first
    parser.add_argument(
        "--conf", "-c", type=open, action=LoadFromFile, help="Configuration yaml file"
    )  # keep second

    # General
    parser.add_argument("-relative_pose", "--relative_pose", default=False, action='store_true')

    parser.add_argument("-i", "--id", type=int, default=0)
    parser.add_argument("-g", "--gpus", default=1, type=int)
    parser.add_argument("-e", "--num_epochs", default=300, type=int)
    parser.add_argument("--eval_freq", default=1.0, type=float)
    parser.add_argument("--precision", default=32, type=int)
    parser.add_argument("-b", "--batch_size", default=256, type=int)
    parser.add_argument("--lr", default=5e-4, type=float)
    parser.add_argument("--gamma", default=0.975, type=float)
    parser.add_argument("--grad_clip_val", default=10.0, type=float)
    parser.add_argument("--exp_scheduler", default=False, action="store_true")
    parser.add_argument("--lr-frequency", default=5, type=int)
    parser.add_argument("--detect_anomaly", default=False, action="store_true")
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--lr-patience", default=10, type=int)
    parser.add_argument("--lr-cooldown", default=10, type=int)
    parser.add_argument("--lr-factor", default=0.75, type=float)
    parser.add_argument("--weight-decay", default=0.0, type=float)
    parser.add_argument("--ema-decay", default=0.999, type=float)
    parser.add_argument("--load-ckpt", default="", type=str)
    parser.add_argument(
        "--log-dir",
        "-l",
        default="/workspace7/e3mol/qm9_logs",
        help="log file",
    )
    parser.add_argument(
        "--num-workers", type=int, default=4, help="Number of workers for data prefetch"
    )

    # Dataset
    parser.add_argument(
        "--dataset", default="qm9", choices=["qm9", "geom_drugs", "geom_qm9"]
    )
    parser.add_argument("--dataset-arg", default={"label": "energy_U0"})
    parser.add_argument("--dataset-root", default="/workspace7/e3mol/qm9_data")
    parser.add_argument("--seed", type=int, default=1, help="random seed (default: 1)")
    parser.add_argument("--accum_batch", default=None, type=int)
    parser.add_argument(
        "--splits", default=None, help="Npz with splits idx_train, idx_val, idx_test"
    )
    parser.add_argument("--inference-batch-size", default=128, type=int)
    parser.add_argument(
        "--train-size",
        type=number,
        default=0.8,
        help="Percentage/number of samples in training set (None to use all remaining samples)",
    )
    parser.add_argument(
        "--val-size",
        type=number,
        default=0.1,
        help="Percentage/number of samples in validation set (None to use all remaining samples)",
    )
    parser.add_argument(
        "--test-size",
        type=number,
        default=0.1,
        help="Percentage/number of samples in test set (None to use all remaining samples)",
    )

    # Model architecture
    parser.add_argument(
        "--model",
        type=str,
        default="equivariant-transformer",
        choices=["eqgat", "equivariant-transformer"],
        help="Which model to train",
    )
    parser.add_argument("--fully-connected", default=False, action="store_true")
    parser.add_argument(
        "--fully-connected-layer",
        default=False,
        type=bool,
        help="Use full adj. layer in MPNN.",
    )
    parser.add_argument(
        "--local-global",
        default=False,
        type=bool,
        help="Use combination of local and global model.",
    )
    parser.add_argument(
        "--use-local-edge-attr",
        default=False,
        type=bool,
        help="Use local edge attributes for bond model.",
    )
    parser.add_argument(
        "--skip-connect-context",
        default=False,
        type=bool,
        help="Use full adj. layer in MPNN.",
    )
    parser.add_argument(
        "--max-num-neighbors",
        type=int,
        default=100,
        help="Maximum number of neighbors to consider in the network",
    )
    parser.add_argument(
        "--cutoff-lower", type=float, default=0.0, help="Lower cutoff in model"
    )
    parser.add_argument(
        "--cutoff-upper", type=float, default=5.0, help="Upper cutoff in model"
    )
    parser.add_argument(
        "--cutoff-lower-global", type=float, default=5.0, help="Lower cutoff in model"
    )
    parser.add_argument(
        "--cutoff-upper-global", type=float, default=10.0, help="Upper cutoff in model"
    )

    # ET args
    parser.add_argument(
        "--embedding-dimension", type=int, default=256, help="Embedding dimension"
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=6,
        help="Number of interaction layers in the model",
    )
    parser.add_argument(
        "--num-rbf",
        type=int,
        default=64,
        help="Number of radial basis functions in model",
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="silu",
        #choices=list(act_class_mapping.keys()),
        help="Activation function",
    )
    parser.add_argument(
        "--rbf-type",
        type=str,
        default="expnorm",
        #choices=list(rbf_class_mapping.keys()),
        help="Type of distance expansion",
    )
    parser.add_argument(
        "--trainable-rbf",
        type=bool,
        default=False,
        help="If distance expansion functions should be trainable",
    )
    parser.add_argument(
        "--neighbor-embedding",
        type=bool,
        default=False,
        help="If a neighbor embedding should be applied before interactions",
    )
    parser.add_argument(
        "--aggr",
        type=str,
        default="add",
        help="Aggregation operation for CFConv filter output. Must be one of 'add', 'mean', or 'max'",
    )
    parser.add_argument(
        "--distance-influence",
        type=str,
        default="both",
        choices=["keys", "values", "both", "none"],
        help="Where distance information is included inside the attention",
    )
    parser.add_argument(
        "--attn-activation",
        default="silu",
        #choices=list(act_class_mapping.keys()),
        help="Attention activation function",
    )
    parser.add_argument(
        "--num-heads", type=int, default=8, help="Number of attention heads"
    )
    parser.add_argument(
        "--derivative",
        default=False,
        type=bool,
        help="If true, take the derivative of the prediction w.r.t coordinates",
    )
    parser.add_argument(
        "--max-z",
        type=int,
        default=100,
        help="Maximum atomic number that fits in the embedding matrix",
    )
    parser.add_argument(
        "--output-model",
        type=str,
        default="VectorCategorical",
        choices=["VectorCategorical"],
        help="The type of output model",
    )

    # EQGAT
    parser.add_argument("--sdim", default=64, type=int)
    parser.add_argument("--latent_dim", default=32, type=int)
    parser.add_argument("--vdim", default=16, type=int)
    parser.add_argument("--edim", default=0, type=int)
    parser.add_argument("--tdim", default=1, type=int)
    parser.add_argument("--omit_norm", default=False, action="store_true")
    parser.add_argument("--use_bond_features", default=False, action="store_true")
    parser.add_argument("--use_all_atom_features", default=False, action="store_true")
    parser.add_argument("--omit_cross_product", default=False, action="store_true")
    parser.add_argument(
        "--use-cross-product",
        type=bool,
        default=True,
        help="Whether or not to perform diffusion",
    )
    parser.add_argument(
        "--use-norm",
        type=bool,
        default=True,
        help="Whether or not to perform diffusion",
    )
    # Diffusion
    parser.add_argument(
        "--diffusion",
        type=bool,
        default=True,
        help="Whether or not to perform diffusion",
    )
    parser.add_argument(
        "--continuous",
        default=False,
        action="store_true",
        help="If the diffusion process is applied on continuous time variable. Defaults to False",
    )
    parser.add_argument(
        "--eps_min", default=1e-3, type=float
    )  # minimum continuous time
    parser.add_argument(
        "--beta_min", default=0.1, type=float
    )  # diffusion coefficients set as if continuous
    parser.add_argument(
        "--beta_max", default=20.0, type=float
    )  # after conversion / T --> 1e-4; 0.02 as originally
    parser.add_argument("--num_diffusion_timesteps", default=300, type=int)
    # diffusion args
    parser.add_argument(
        "--num-atom-types",
        type=int,
        default=5,
        help="Number of unique atom types in the data.",
    )
    parser.add_argument(
        "--atom-types",
        type=list,
        default=[1, 6, 7, 8, 9],
        help="Specify atom types in the dataset here.",
    )
    parser.add_argument(
        "--properties-list",
        type=list,
        default=[],
        help="Single or multiple properties as context for the diffusion model as a list",
    )
    parser.add_argument(
        "--num-context-features",
        type=int,
        default=0,
        help="How many features in the context vector?",
    )
    parser.add_argument(
        "--include-charges",
        type=bool,
        default=False,
        help="Whether or not to use charges besides atom numbers for diffusion prediction",
    )
    parser.add_argument(
        "--remove-hs",
        type=bool,
        default=False,
        help="Whether or not to remove hydrogens for diffusion",
    )
    parser.add_argument(
        "--condition-time",
        type=bool,
        default=False,
        help="Whether or not to condition diffusion on time",
    )
    parser.add_argument("--ode-regularization", type=float, default=1e-3)
    parser.add_argument(
        "--gradient-clipping",
        type=bool,
        default=False,
        help="Whether or not to perform adaptive gradient clipping",
    )
    parser.add_argument(
        "--timesteps", type=int, default=300, help="Number of timesteps for diffusion"
    )
    parser.add_argument(
        "--norm-values",
        type=eval,
        default=[1, 4, 1],
        help="normalize factors for [x, categorical, integer]",
    )
    parser.add_argument(
        "--norm-biases",
        type=eval,
        default=(None, 0.0, 0.0, 0.0),
        help="normalize factors for [x, categorical, integer]",
    )
    parser.add_argument(
        "--n-stability-samples",
        type=int,
        default=1000,
        help="Number of samples to compute the stability",
    )
    parser.add_argument(
        "--test-interval",
        type=int,
        default=10,
        help="Number of samples to compute the stability",
    )
    parser.add_argument(
        "--noise-schedule",
        type=str,
        default="polynomial_2",
        choices=["polynomial_2", "learned"],
        help="Learned or fixed noise scheduler.",
    )
    parser.add_argument(
        "--noise-precision", type=float, default=1.0e-5, help="Noise precision."
    )
    parser.add_argument(
        "--loss-type",
        type=str,
        default="l2",
        choices=["l2", "vlb"],
        help="Which loss to use.",
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        default="qm9",
        choices=["qm9", "drugs"],
        help="Which experiment dataset to use.",
    )
    parser.add_argument(
        "--use-rbf",
        type=bool,
        default=True,
        help="Whether or not to use RBF expansions in EQGAT",
    )
    parser.add_argument("--parametrization", default="eps", type=str)
    parser.add_argument("--energy_preserving", default=False, action="store_true")
    parser.add_argument(
        "--max_num_conformers",
        default=30,
        type=int,
        help="Maximum number of conformers per molecule. \
                            Defaults to 30. Set to -1 for all conformers available in database",
    )
    return parser
