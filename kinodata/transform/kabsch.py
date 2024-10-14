import torch
from torch import Tensor


@torch.no_grad()
def kabsch_alignment(P: Tensor, Q: Tensor) -> Tensor:
    assert P.size() == Q.size(), (P.size(), Q.size())
    p_mean = P.mean(0)
    q_mean = Q.mean(0)

    P_c = P - p_mean
    Q_c = Q - q_mean

    H = P_c.t() @ Q_c
    U, S, Vh = torch.linalg.svd(H)
    V = Vh.mH
    R = V @ U.t()
    t = q_mean - (R @ p_mean).t()

    return P @ R.t() + t


def pad_right(padding_target: Tensor, padding_source: Tensor, dim: int = 0):
    # pad the target to the right with values from source
    # pad_right([1,2,3], [4,5,6,7,8]) will return [1,2,3,7,8]
    offset = padding_target.size(dim) - padding_source.size(dim)
    assert offset < 0, "source size must be greater than target size"
    padding = padding_source[offset:]
    return torch.cat((padding_target, padding), 0)


def pseudo_kabsch_alignment(P: Tensor, Q: Tensor) -> Tensor:
    assert P.size(1) == Q.size(1)
    p_size_orig = P.size(0)
    if P.size(0) == Q.size(0):
        return kabsch_alignment(P, Q)
    if p_size_orig > Q.size(0):
        Q = pad_right(Q, P)
    else:
        P = pad_right(P, Q)

    P_aligned = kabsch_alignment(P, Q)
    return P_aligned[:p_size_orig]


if __name__ == "__main__":
    P = torch.randn(2, 3)
    Q = torch.randn(4, 3)

    P_ = pseudo_kabsch_alignment(P, Q)
    assert P.size() == P_.size()
