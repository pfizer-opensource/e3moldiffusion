import torch
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import HeteroData

from ..types import NodeType, RelationType


class TransformToComplexGraph(BaseTransform):
    def __init__(self, remove_heterogeneous_representation: bool = False):
        self.remove_heterogeneous_representation = remove_heterogeneous_representation

    def __call__(self, data: HeteroData) -> HeteroData:
        ligand_store = data[NodeType.Ligand]
        pocket_store = data[NodeType.Pocket]
        ligand_edge_store = data[
            NodeType.Ligand, RelationType.Covalent, NodeType.Ligand
        ]
        pocket_edge_store = data[
            NodeType.Pocket, RelationType.Covalent, NodeType.Pocket
        ]

        x = torch.cat((pocket_store.x, ligand_store.x), dim=0)
        z = torch.cat((pocket_store.z, ligand_store.z), dim=0)
        pos = torch.cat((pocket_store.pos, ligand_store.pos), dim=0)
        edge_index = torch.cat(
            (
                pocket_edge_store.edge_index,
                ligand_edge_store.edge_index + pocket_store.x.size(0),
            ),
            dim=1,
        )
        edge_attr = torch.cat(
            (
                pocket_edge_store.edge_attr,
                ligand_edge_store.edge_attr,
            ),
            dim=0,
        )

        data[NodeType.Complex].x = x
        data[NodeType.Complex].z = z
        data[NodeType.Complex].pos = pos
        data[
            NodeType.Complex, RelationType.Covalent, NodeType.Complex
        ].edge_index = edge_index
        data[
            NodeType.Complex, RelationType.Covalent, NodeType.Complex
        ].edge_attr = edge_attr

        if self.remove_heterogeneous_representation:
            del data[NodeType.Ligand]
            del data[NodeType.Pocket]
            del data[NodeType.Ligand, RelationType.Covalent, NodeType.Ligand]
            del data[NodeType.Pocket, RelationType.Covalent, NodeType.Pocket]

        return data
