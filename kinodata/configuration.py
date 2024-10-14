import inspect

from typing import (
    Any,
    Dict,
    Hashable,
    Iterable,
    MutableMapping,
    Optional,
    Union,
    TypeVar,
    Callable,
)
import traceback
from collections import defaultdict
from pathlib import Path
from warnings import warn
from argparse import ArgumentParser

import torch
import yaml
import json
import sys

from kinodata.types import (
    INTRAMOL_STRUCTURAL_EDGE_TYPES,
    INTERMOL_STRUCTURAL_EDGE_TYPES,
)

T = TypeVar("T")

_ROOT = Path(__file__).parents[1]


def _find_default_config_file():
    try:
        idx = sys.argv.index("--config")
        fp = Path(sys.argv[idx + 1])
        return fp
    except ValueError:
        fp = _ROOT
        yaml_files = list(fp.glob("*.yaml"))
        if len(yaml_files) == 1:
            return yaml_files[0]
    return None


class Config(dict):
    def __getattribute__(self, __name: str) -> Any:
        if __name in self:
            return self[__name]
        return super().__getattribute__(__name)

    def __setattr__(self, __name: str, __value: Any) -> None:
        self[__name] == __value

    def intersect(self, other: MutableMapping) -> "Config":
        return Config({k: v for k, v in self.items() if k in other})

    def subset(self, keys: Iterable[Hashable]) -> "Config":
        return Config({key: self[key] for key in keys if key in self})

    def update(self, other: MutableMapping, allow_duplicates: bool = True) -> "Config":  # type: ignore
        if not allow_duplicates:
            intersection = self.intersect(other)
            if len(intersection) > 0:
                raise ValueError(f"Duplicate keys detected: {intersection.keys()}")
        super().update(other)
        return self

    def init(self, obj: Union[type[T], Callable[..., T]], *args, **kwargs) -> T:
        obj_signature = inspect.signature(obj)
        sub_config = self.subset(obj_signature.parameters.keys())
        sub_config.update(kwargs, allow_duplicates=False)
        bound_arguments = obj_signature.bind(*args, **sub_config)
        return obj(*bound_arguments.args, **bound_arguments.kwargs)

    def argparser(
        self,
        admissible_types: list = [int, float, str, Path],
        overwrite_default_values: bool = True,
    ) -> ArgumentParser:
        parser = ArgumentParser()
        for key, value in self.items():
            if not any(isinstance(value, t) for t in admissible_types):
                continue
            default = None if overwrite_default_values else value
            parser.add_argument(f"--{key}", default=default, type=type(value))
        return parser

    def update_from_args(self, known_only: bool = True) -> "Config":
        if not known_only:
            raise NotImplementedError
        parser = self.argparser(overwrite_default_values=True)
        args, unknown = parser.parse_known_args()
        shared_args = {key: getattr(args, key) for key in self if hasattr(args, key)}
        updated_args = {
            key: value for key, value in shared_args.items() if value is not None
        }
        print(f"Updating config from args: {updated_args}")
        return self.update(updated_args)

    def update_from_file(
        self,
        fp: Union[str, Path, None] = None,
        verbose: bool = True,
    ) -> "Config":
        if fp is None:
            fp = _find_default_config_file()
        if fp is None:
            warn("Unable to find a config file")
            return self
        fp = Path(fp)
        if not fp.exists():
            warn(f"Config file does not exist: {fp}")
            return self
        file_config = load_from_file(fp)
        if verbose:
            print(f"Reading additional config from {fp}:")
            print("\n".join([f"{key}: {value}" for key, value in file_config.items()]))
        return self.update(file_config)

    def __repr__(self) -> str:
        inner = ", ".join([f"{key}={value}" for key, value in self.items()])
        return f"{self.__class__.__name__}({inner})"


configs: Dict[str, Config] = dict()


def register(
    config_name: str, config: Optional[Config] = None, **kwargs: Any
) -> Config:
    if config is None:
        config = Config(**kwargs)
    configs[config_name] = config.update(kwargs)
    return config


def extend(config_name: str, **kwargs: Any):
    configs[config_name] |= kwargs


def load_from_file(fp: Path) -> Config:
    try:
        file_stream = fp.read_text()
        config = yaml.full_load(file_stream)
    except:
        print(f"Not a valid config yaml file: {fp}")
        traceback.print_exc()
        exit(1)
    return config


def from_wandb(run=None) -> Config:
    if run is not None:
        dict_config = {
            key: obj["value"] for key, obj in json.loads(run.json_config).items()
        }
        return Config(dict_config)
    raise ValueError()


def get(*config_names: str) -> Config:
    key_counts: Dict[str, int] = defaultdict(int)
    for name in config_names:
        assert name in configs, f'Unknown config: "{name}"'
        for key in configs[name]:
            key_counts[key] += 1

    duplicate_keys = [key for key in key_counts if key_counts[key] > 1]
    if len(duplicate_keys) > 0:
        raise ValueError(f"Duplicate config keys: {duplicate_keys}")

    config = Config()
    for name in config_names:
        config = config.update(configs[name])
    return config


register(
    "data",
    interaction_radius=5.0,
    residue_interaction_radius=12.0,  # deprecated
    node_types=["ligand", "pocket"],
    edge_types=INTRAMOL_STRUCTURAL_EDGE_TYPES[:1] + INTERMOL_STRUCTURAL_EDGE_TYPES,
    seed=420,
    use_bonds=True,
    data_split=None,  # deprecated
    need_distances=True,
    num_residue_features=6,  # deprecated
    additional_atom_features=False,
    remove_hydrogen=True,
    filter_rmsd_max_value=2.0,
    split_type="pocket-k-fold",
    split_index=0,
    k_fold=5,
    perturb_ligand_positions=0.0,
    perturb_pocket_positions=0.0,
    perturb_complex_positions=0.0,
    add_docking_scores=False,
)

# deprecated
register(
    "egnn",
    model_type="egnn",
    num_mp_layers=3,
    hidden_channels=128,
    act="silu",
    final_act="softplus",
    mp_type="rbf",
    mp_reduce="sum",
    rbf_size=32,
    readout_aggregation_type="sum",
)

register(
    "training",
    optim="adamw",
    lr=1e-4,
    weight_decay=3e-6,
    dropout=0.0,
    batch_size=128,
    accumulate_grad_batches=1,
    epochs=300,
    num_workers=32 if torch.cuda.is_available() else 0,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    loss_type="mse",
    lr_factor=0.9,
    lr_patience=8,
    early_stopping_patience=24,
    min_lr=1e-6,
    clip_grad_value=None,
    dry_run=False,
)
