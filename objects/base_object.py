import torch
from vpython import *
from typing import Union, Iterable

from utils import convert_to_vector

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

        self._pos = convert_to_vector(pos)
        self._pos_before = vector(self._pos)
        self._dir = convert_to_vector(dir)
        self._dir_before = vector(self._dir)
        
        self._mass = mass
        self._color = col
        self._pos_fixed = pos_fixed
        self._rot_fixed = rot_fixed
        self._index = simul.add_object(self)
        self._obj: standardAttributes = None
    
    # Spatial properties
    @property
    def pos(self) -> vector:
        return self._pos
    
    @property
    def pos_tensor(self) -> torch.Tensor:
        return torch.as_tensor(self._pos.value)

    @property
    def dir(self) -> vector:
        return self._dir
    
    @property
    def dir_tensor(self) -> torch.Tensor:
        return torch.as_tensor(self._dir.value)
    
    @property
    def lin_vel(self) -> vector:
        return (self._pos - self._pos_before) / self._simul.h
    
    @property
    def lin_vel_tensor(self) -> torch.Tensor:
        return torch.as_tensor(self.lin_vel.value)
    
    @property
    def ang_vel(self) -> vector:
        return (self._dir - self._dir_before) / self._simul.h
    
    @property
    def ang_vel_tensor(self) -> vector:
        return torch.as_tensor(self.ang_vel.value)
    
    @property
    def pos_fixed(self) -> bool:
        return self._pos_fixed

    @property
    def rot_fixed(self) -> bool:
        return self._rot_fixed
    
    @pos.setter
    def pos(self, value: vector) -> None:
        self._pos_before = vector(self._pos)
        self._pos = vector(value)
        self._obj.pos = self._pos
    
    @dir.setter
    def dir(self, value: vector) -> None:
        self._dir_before = vector(self._dir)
        self._dir = vector(value)
        self._obj.up = vector(*self.rot_mat[:, 1])
        self._obj.axis = vector(*self.rot_mat[:, 0])

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
        return torch.concatenate([self.pos_tensor, self.dir_tensor])
    
    @property
    def v(self) -> torch.Tensor:
        return torch.concatenate([self.lin_vel_tensor, self.ang_vel_tensor])
    
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

        x, y, z = self._dir.value
        return torch.as_tensor([
            [0, cos(x),  sin(y) * sin(x)],
            [0, sin(x), -sin(y) * cos(x)],
            [1,      0,           cos(y)],
        ])
    
    @property
    def rot_mat(self) -> torch.Tensor:
        x, y, z = self._dir.value
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

        self.pos = vector(*_pos)
        self.dir = vector(*_dir)