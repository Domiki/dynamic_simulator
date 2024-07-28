import torch
from vpython import *
from typing import Union, Iterable

from utils import convert_to_tensor, convert_to_vector

class BaseObject:
    def __init__(self,
                 simul,
                 pos: Union[vector, Iterable]=(0, 0, 0),
                 dir: Union[vector, Iterable]=(0, 0, 0),
                 mass: float=1,
                 col: color=color.white,
                 pos_fixed: bool=False,
                 rot_fixed: bool=False):
        self._simul = simul

        self._pos = convert_to_tensor(pos)
        self._pos_before = self._pos.clone()
        self._dir = convert_to_tensor(dir)
        self._dir_before = self._dir.clone()
        
        self._mass = mass
        self._color = col
        self._pos_fixed = pos_fixed
        self._rot_fixed = rot_fixed
        self._index = simul.add_object(self)
        self._obj: standardAttributes = None
    
    # Spatial properties
    @property
    def pos(self) -> torch.Tensor:
        return self._pos

    @property
    def dir(self) -> torch.Tensor:
        return self._dir
    
    @property
    def lin_vel(self) -> torch.Tensor:
        return (self._pos - self._pos_before) / self._simul.h
    
    @property
    def ang_vel(self) -> torch.Tensor:
        return (self._dir - self._dir_before) / self._simul.h
    
    @property
    def pos_fixed(self) -> bool:
        return self._pos_fixed

    @property
    def rot_fixed(self) -> bool:
        return self._rot_fixed
    
    @pos.setter
    def pos(self, value: Union[vector, Iterable]) -> None:
        self._pos_before = self._pos.clone()
        self._pos = convert_to_tensor(value)
        self._obj.pos = convert_to_vector(self._pos)
    
    @dir.setter
    def dir(self, value: Union[vector, Iterable]) -> None:
        self._dir_before = self._dir.clone()
        self._dir = convert_to_tensor(value)
        self._obj.up = convert_to_vector(self.rot_mat[:, 1])
        self._obj.axis = convert_to_vector(self.rot_mat[:, 0])

    # Object properties
    @property
    def mass(self) -> float:
        return self._mass
    
    @property
    def color(self) -> color:
        return self._color
    
    @property
    def index(self) -> int:
        return self._index

    @property
    def obj(self) -> standardAttributes:
        return self._obj
    
    # Dynamic properties
    @property
    def q(self) -> torch.Tensor:
        return torch.concatenate([self.pos, self.dir])
    
    @property
    def v(self) -> torch.Tensor:
        return torch.concatenate([self.lin_vel, self.ang_vel])
    
    # Matrix porperties
    @property
    def lin_vel_coeff_mat(self) -> torch.Tensor:
        if self._pos_fixed:
            return torch.zeros((3, 3))

        return torch.eye(3)
    
    @property
    def ang_vel_coeff_mat(self) -> torch.Tensor:
        if self._rot_fixed:
            return torch.zeros((3, 3))

        x, y, z = self._dir
        return torch.as_tensor([
            [0, cos(x),  sin(y) * sin(x)],
            [0, sin(x), -sin(y) * cos(x)],
            [1,      0,           cos(y)],
        ])
    
    @property
    def rot_mat(self) -> torch.Tensor:
        x, y, z = self._dir
        return torch.as_tensor([
            [
                 cos(z) * cos(x) - cos(y) * sin(x) * sin(z),
                -sin(z) * cos(x) - cos(y) * sin(x) * cos(z),
                 sin(y) * sin(x)
            ],
            [
                 cos(z) * sin(x) + cos(y) * cos(x) * sin(z),
                -sin(z) * sin(x) + cos(y) * cos(x) * cos(z),
                -sin(y) * cos(x)
            ],
            [
                sin(y) * sin(z),
                sin(y) * cos(z),
                cos(y)
            ],
        ])

    # Update function
    def update(self, q: torch.Tensor) -> None:
        _pos = q[0:3]
        _dir = q[3:6]

        self.pos = convert_to_vector(_pos)
        self.dir = convert_to_vector(_dir)