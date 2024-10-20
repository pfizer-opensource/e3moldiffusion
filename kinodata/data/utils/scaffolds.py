from typing import Iterator, Protocol, Generic, TypeVar

import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


IdType = TypeVar("IdType")


class IdentifiableSmiles(Protocol, Generic[IdType]):
    ident: IdType
    smiles: str


class SmilesDataset(Protocol, Generic[IdType]):
    def __getitem__(self, index) -> IdentifiableSmiles[IdType]:
        ...

    def __iter__(self) -> Iterator[IdentifiableSmiles[IdType]]:
        ...


def mol_to_scaffold(mol) -> str:
    mol_scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    mol_scaffold_generic = MurckoScaffold.MakeScaffoldGeneric(mol_scaffold)
    smiles_scaffold_generic = Chem.CanonSmiles(Chem.MolToSmiles(mol_scaffold_generic))
    return smiles_scaffold_generic


def generate_scaffolds(
    smiles_dataset: SmilesDataset[IdType],
) -> pd.DataFrame:
    """
    https://www.blopig.com/blog/2021/06/out-of-distribution-generalisation-and-scaffold-splitting-in-molecular-property-prediction/

    Parameters
    ----------
    dataset : SmilesDataset[IdType]
        _description_

    Returns
    -------
    pd.DataFrame
        _description_
    """

    scaffolds = dict()
    for data_item in smiles_dataset:
        mol = Chem.MolFromSmiles(data_item.smiles)
        smiles_scaffold_generic = mol_to_scaffold(mol)
        scaffolds[data_item.ident] = smiles_scaffold_generic

    scaffold_dataset = pd.DataFrame(
        {"ident": list(scaffolds.keys()), "scaffold": list(scaffolds.values())}
    )

    return scaffold_dataset
