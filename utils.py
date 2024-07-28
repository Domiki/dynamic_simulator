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
        return torch.Tensor(vec.value)
    else:
        return torch.Tensor(vec)

def convert_to_vector(value: Iterable) -> vector:
    assert len(value) == 3
    return vector(*value)