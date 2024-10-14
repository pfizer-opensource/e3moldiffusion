from typing import List, Sequence, Callable
from enum import Enum

from torch_geometric.data import Data


class ActivityTypes(Enum):
    pic50 = "pIC50"
    pkd = "pKd"
    pki = "pKi"


class FilterActivityType:
    def __init__(self, allowed: Sequence[ActivityTypes]):
        self.allowed = set(at.value for at in allowed)

    def __call__(self, data: Data) -> bool:
        return data.activity_type in self.allowed

    def __repr__(self) -> str:
        return f"FilterActivityType(allowed={''.join(self.allowed)})"


class FilterActivityScore:
    """
    Filter an activity by its `docking_score` attribute using `threshold` as an upper threshold.
    """

    def __init__(self, threshold: float = 0):
        self.threshold = threshold

    def __call__(self, data: Data) -> bool:
        return data.docking_score < self.threshold

    def __repr__(self) -> str:
        return f"FilterActivityScore(threshold={self.threshold})"


class FilterCombine:
    """
    Compose multiple activity filters using 'and'.
    """

    def __init__(self, filters: List[Callable]):
        self.filters = filters

    def __call__(self, data: Data) -> bool:
        return all(f(data) for f in self.filters)

    def __repr__(self) -> str:
        return repr(self.filters)
