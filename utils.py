import torch
from typing import Union, Iterable
from vpython import vector

def skew_matrix(vec: torch.Tensor) -> torch.Tensor:
    x, y, z = vec
    return torch.as_tensor([
        [0, -z, y],
        [z, 0, -x],
        [-y, x, 0],
    ])

def convert_to_tensor(vec: Union[vector, Iterable]) -> torch.Tensor:
    if isinstance(vec, vector):
        return torch.as_tensor(vec.value, dtype=torch.float64)
    else:
        return torch.as_tensor(vec, dtype=torch.float64)

def convert_to_vector(value: Iterable) -> vector:
    assert len(value) == 3
    return vector(*value)

def solve(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    P, L, U = torch.linalg.lu(A, pivot=False)

    diag_U = torch.diag(U)
    zeros = torch.zeros_like(diag_U)
    non_singular_rows = (~torch.isclose(diag_U, zeros)).nonzero().squeeze()
    A = A[non_singular_rows][:, non_singular_rows]
    B = B[non_singular_rows]

    return torch.squeeze(torch.linalg.solve(A, B))