import os

DEFAULT_SAVE_DIR = os.path.join(os.getcwd(), "saves_diffusion0")

if not os.path.exists(DEFAULT_SAVE_DIR):
    os.makedirs(DEFAULT_SAVE_DIR)


def add_arguments(parser):
    """Helper function to fill the parser object.

    Args:
        parser: Parser object
    Returns:
        parser: Updated parser object
    """

    # GENERAL
    parser.add_argument("-i", "--id", type=int, default=0)
    parser.add_argument("-g", "--gpus", default=1, type=int)
    parser.add_argument("-e", "--num_epochs", default=300, type=int)
    parser.add_argument("--eval_freq", default=0.8, type=float)
    parser.add_argument("-s", "--save_dir", default=DEFAULT_SAVE_DIR, type=str)
    parser.add_argument("--precision", default=32, type=int)
    parser.add_argument("-b", "--batch_size", default=256, type=int)
    parser.add_argument("--lr", default=5e-4, type=float)
    parser.add_argument("--gamma", default=0.975, type=float)
    parser.add_argument("--grad_clip_val", default=10.0, type=float)
    parser.add_argument("--exp_scheduler", default=False, action="store_true")
    parser.add_argument("--frequency", default=5, type=int)
    parser.add_argument("--detect_anomaly", default=False, action="store_true")
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--patience", default=10, type=int)
    parser.add_argument("--cooldown", default=10, type=int)
    parser.add_argument("--factor", default=0.75, type=float)

    parser.add_argument("--load_ckpt", default="", type=str)
    parser.add_argument("--dataset", default="drugs", choices=["qm9", "drugs"])
    #parser.add_argument(
    #    "--subset_frac",
    #    default=0.1,
    #    type=float,
    #    help="Fraction to train on only for experimenting. Defaults to 0.1",
    #)
    parser.add_argument("--max_num_conformers", default=30, type=int,
                        help="Maximum number of conformers per molecule. \
                            Defaults to 30. Set to -1 for all conformers available in database"
                            )

    parser.add_argument("--accum_batch", default=None, type=int)

    parser.add_argument("--energy_preserving", default=False, action="store_true")
    parser.add_argument("--sdim", default=64, type=int)
    parser.add_argument("--vdim", default=16, type=int)
    parser.add_argument("--edim", default=0, type=int)
    parser.add_argument("--tdim", default=16, type=int)
    parser.add_argument("--num_layers", default=4, type=int)
    parser.add_argument("--omit_norm", default=False, action="store_true")
    parser.add_argument("--fully_connected", default=False, action="store_true")
    parser.add_argument("--cutoff", default=5.0, type=float)

    parser.add_argument(
        "--eps_min", default=1e-3, type=float
    )  # minimum continuous time
    parser.add_argument(
        "--beta_min", default=0.1, type=float
    )  # diffusion coefficients set as if continuous
    parser.add_argument(
        "--beta_max", default=20.0, type=float
    )  # after conversion / T --> 1e-4; 0.02 as originally
    parser.add_argument("--num_diffusion_timesteps", default=1000, type=int)

    return parser