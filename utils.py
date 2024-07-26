import torch
from typing import Union, Iterable
from vpython import vector

def skew_matrix(vec: Union[vector, Iterable]) -> torch.Tensor:
    vec = convert_to_vector(vec)
    return torch.as_tensor([
        [0, -vec.z, vec.y],
        [vec.z, 0, -vec.x],
        [-vec.y, vec.x, 0],
    ])

def convert_to_vector(vec: Union[vector, Iterable]) -> vector:
    if isinstance(vec, vector):
        return vector(vec)
    else:
        assert len(vec) == 3
        return vector(*vec)