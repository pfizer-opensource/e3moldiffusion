from functools import partial
from typing import Any, Optional
from torch_geometric.data import HeteroData
from torch_geometric.transforms import BaseTransform

import operator


def _get_metadata(data: HeteroData, metadata_key: str, default=None):
    try:
        return getattr(data, metadata_key)
    except AttributeError:
        return default


class MetadataFilter(BaseTransform):
    def __init__(
        self,
        key: str,
        exact_value: Optional[Any] = None,
        min_value: Optional[Any] = None,
        max_value: Optional[Any] = None,
    ) -> None:
        super().__init__()
        self._repr = "Filter({key}, {operators})"
        conditions = []
        if exact_value is not None:
            conditions.append((operator.eq, exact_value))
        if min_value is not None:
            conditions.append((operator.ge, min_value))
        if max_value is not None:
            conditions.append((operator.le, max_value))

        self._repr = self._repr.format(
            key=key,
            operators=", ".join([f"{op.__name__}{val:.2f}" for op, val in conditions]),
        )

        def _filter(data):
            value = _get_metadata(data, key)
            if value is None:
                raise ValueError(f"Unable to retrieve metdata for key {key}")
            return all(op(value, cmp) for op, cmp in conditions)

        self.filter = _filter

    def __repr__(self):
        return self._repr

    def __call__(self, data: Any) -> Any:
        return self.filter(data)


def FilterDockingRMSD(maximum_rmsd: float):
    return MetadataFilter("predicted_rmsd", max_value=maximum_rmsd)
