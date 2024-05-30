from experiments.utils import LoadFromFile


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

    parser.add_argument(
        "--dataset",
        default="drugs",
        choices=[
            "qm9",
            "drugs",
            "aqm",
            "aqm_qm7x",
            "pcqm4mv2",
            "pepconf",
            "crossdocked",
        ],
    )
    parser.add_argument(
        "--dataset-root", default="/hpfs/userws/cremej01/projects/data/geom"
    )
    parser.add_argument("--num-workers", default=4, type=int)

    parser.add_argument("--use-adaptive-loader", default=False, action="store_true")
    parser.add_argument("--remove-hs", default=False, action="store_true")
    parser.add_argument("--select-train-subset", default=False, action="store_true")
    parser.add_argument("--train-size", default=0.8, type=float)
    parser.add_argument("--val-size", default=0.1, type=float)
    parser.add_argument("--test-size", default=0.1, type=float)
    parser.add_argument("-b", "--batch-size", default=32, type=int)

    parser.add_argument(
        "--save-dir", default="/hpfs/userws/cremej01/projects/data/geom"
    )
    parser.add_argument(
        "--sdf-path", default="/hpfs/userws/cremej01/projects/data/geom"
    )

    parser.add_argument("--num-bond-classes", default=5, type=int)
    parser.add_argument("--num-charge-classes", default=6, type=int)

    return parser
