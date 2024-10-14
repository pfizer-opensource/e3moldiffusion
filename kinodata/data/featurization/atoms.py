from typing import Callable, Dict, Generic, List, Optional, Sequence, TypeVar
import numpy as np
from rdkit.Chem import rdchem

CT = TypeVar("CT")


class AtomFeaturizer:
    @property
    def size(self) -> int:
        return 0

    def compute(self, mol) -> np.ndarray:
        ...


class OneHotFeaturizer(AtomFeaturizer, Generic[CT]):
    def __init__(
        self,
        categories: Sequence[CT],
        extractor: Callable[[rdchem.Atom], CT],
        name: Optional[str] = None,
    ):
        self.__mapping = {category: index for index, category in enumerate(categories)}
        self.__extractor = extractor
        self.__name = self.__class__.__name__ if name is None else name

    @property
    def mapping(self) -> Dict[CT, int]:
        return self.__mapping

    @property
    def size(self) -> int:
        return len(self.__mapping) + 1

    def get_category_index(self, atom) -> int:
        try:
            cat = self.__extractor(atom)
            return self.__mapping[cat]
        except (KeyError, ValueError):
            # default category
            return self.size - 1

    def compute(self, mol) -> np.ndarray:
        result = np.zeros((mol.GetNumAtoms(), self.size))
        for i, atom in enumerate(mol.GetAtoms()):
            k = self.get_category_index(atom)
            result[i, k] = 1.0
        return result

    def __repr__(self) -> str:
        return f"{self.__name}({self.__mapping})"


class ComposeOneHot(AtomFeaturizer):
    def __init__(self, oneh_hot_featurizers: List[OneHotFeaturizer]) -> None:
        super().__init__()
        self.__one_hot_featurizers = oneh_hot_featurizers

    @property
    def size(self) -> int:
        return sum(f.size for f in self.__one_hot_featurizers)

    def compute(self, mol) -> np.ndarray:
        result = np.zeros((mol.GetNumAtoms(), self.size))
        offsets = np.cumsum(
            np.array([0] + [f.size for f in self.__one_hot_featurizers])
        )[:-1]
        for i, atom in enumerate(mol.GetAtoms()):
            k = np.array(
                [f.get_category_index(atom) for f in self.__one_hot_featurizers]
            )
            result[i, k + offsets] = 1.0
        return result

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({', '.join([repr(f) for f in self.__one_hot_featurizers])})"


FormalCharge = OneHotFeaturizer(
    [-2, -1, 0, 1, 2], rdchem.Atom.GetFormalCharge, name="FormalCharge"
)
NumHydrogens = OneHotFeaturizer(
    [0, 1, 2, 3, 4], rdchem.Atom.GetTotalNumHs, name="NumHydrogens"
)
AtomFeatures = ComposeOneHot([FormalCharge, NumHydrogens])
