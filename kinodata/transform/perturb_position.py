import torch

from torch import Tensor
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import HeteroData

from kinodata.transform.kabsch import kabsch_alignment


def rmd(pos1: Tensor, pos2: Tensor, dim: int = 0, feature_dim: int = 1) -> Tensor:
    assert pos1.size() == pos1.size()
    dist = (pos1 - pos2).pow(2).sum(dim=feature_dim, keepdim=True).sqrt()
    return dist.mean(dim)


def rmsd(pos1: Tensor, pos2: Tensor, dim: int = 0, feature_dim: int = 1) -> Tensor:
    assert pos1.size() == pos1.size()
    square_dist = (pos1 - pos2).pow(2).sum(dim=feature_dim, keepdim=True)
    return square_dist.mean(dim).sqrt()


class PerturbAtomPositions(BaseTransform):
    def perturb(self, pos: Tensor, noise: Tensor) -> Tensor:
        noisy_pos = pos + noise
        if self.re_align:
            noisy_pos = kabsch_alignment(noisy_pos, pos)
        return noisy_pos

    def perturb_gaussian(self, pos: Tensor, mean: float = 0, std: float = 1) -> Tensor:
        noise = torch.empty_like(pos).normal_(mean, std, generator=self.generator)
        return self.perturb(pos, noise)

    def __init__(
        self,
        atom_key: str,
        std: float = 1,
        overwrite_pos: bool = True,
        store_rmsd: bool = False,
        re_align: bool = False,
        seed: int = 0,
    ) -> None:
        self.atom_key = atom_key
        self.std = std
        self.overwrite_pos = overwrite_pos
        self.store_rmsd = store_rmsd
        self.re_align = re_align

        self.store_key = "pos" if self.overwrite_pos else "noisy_pos"
        self.generator = torch.Generator().manual_seed(seed)

    def __call__(self, data: HeteroData) -> HeteroData:
        pos = data[self.atom_key].pos
        noisy_pos = self.perturb_gaussian(pos, std=self.std)

        if self.store_rmsd:
            data[self.atom_key].noisy_pos_rmsd = rmsd(pos, noisy_pos)

        setattr(data[self.atom_key], self.store_key, noisy_pos)
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.atom_key}, {self.std:.2f})"


if __name__ == "__main__":
    perturb = PerturbAtomPositions("...")

    pos = torch.rand(100, 3)
    new_pos = perturb.perturb_gaussian(pos, 0, std=0.3)

    print(rmd(pos, new_pos), rmsd(pos, new_pos))
