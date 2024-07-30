import torch
from typing import Union, Iterable
from vpython import vector

# convert functions
def convert_to_tensor(vec: Union[vector, Iterable]) -> torch.Tensor:
    if isinstance(vec, vector):
        return torch.as_tensor(vec.value, dtype=torch.float64)
    else:
        return torch.as_tensor(vec, dtype=torch.float64)

def convert_to_vector(value: Iterable[float]) -> vector:
    assert len(value) == 3
    return vector(*value)

# matrix functions
def skew_matrix(vec: torch.Tensor) -> torch.Tensor:
    x, y, z = vec
    return torch.as_tensor([
        [0, -z, y],
        [z, 0, -x],
        [-y, x, 0],
    ])

# linalg functions
def get_bases(bases: torch.Tensor) -> torch.Tensor:
    n, dim = bases.size()

    for i in range(n):
        for j in range(i + 1, n):
            assert torch.dot(bases[i], bases[j]) != 0, "Bases not orthogonal"

    A = torch.eye(dim)
    A[:, :n] = bases.T

    Q, R = torch.qr(A)
    return Q.T

def solve(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    P, L, U = torch.linalg.lu(A)

    diag_U = torch.diag(U)
    zeros = torch.zeros_like(diag_U)
    non_singular_rows = (~torch.isclose(diag_U, zeros)).nonzero().squeeze()
    A = A[non_singular_rows][:, non_singular_rows]
    B = B[non_singular_rows]

    return torch.squeeze(torch.linalg.solve(A, B))