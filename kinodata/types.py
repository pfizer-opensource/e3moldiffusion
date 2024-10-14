from typing import Any, Dict, List, Tuple
from torch import Tensor
from torch.utils.data import Subset


class NodeType:
    Ligand = "ligand"
    Pocket = "pocket"
    Complex = "complex"
    PocketResidue = "pocket_residue"


class RelationType:
    Covalent = "bond"
    Interacts = "interacts"  # between different molecules
    Intraacts = "intraacts"  # within a molecule, superset of covalent interactions


Kwargs = Dict[str, Any]
NodeEmbedding = Dict[NodeType, Tensor]
EdgeType = Tuple[NodeType, RelationType, NodeType]
DataSplit = Dict[str, Subset]

INTERMOL_STRUCTURAL_EDGE_TYPES: List[EdgeType] = [
    (NodeType.Ligand, RelationType.Interacts, NodeType.Pocket),
    (NodeType.Pocket, RelationType.Interacts, NodeType.Ligand),
]

INTRAMOL_STRUCTURAL_EDGE_TYPES: List[EdgeType] = [
    (NodeType.Ligand, RelationType.Intraacts, NodeType.Ligand),
    (NodeType.Pocket, RelationType.Intraacts, NodeType.Pocket),
]

COVALENT_EDGE_TYPES: List[EdgeType] = [
    (NodeType.Ligand, RelationType.Covalent, NodeType.Ligand),
    (NodeType.Pocket, RelationType.Covalent, NodeType.Pocket),
]
