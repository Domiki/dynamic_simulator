import torch
from typing import Union, Iterable
from vpython import *

from objects import BaseObject
from joints import BaseJoint
from utils import convert_to_tensor, convert_to_vector, skew_matrix

class BallJoint(BaseJoint):
    def __init__(self,
                 simul,
                 obj1: BaseObject,
                 obj2: BaseObject,
                 pos: Union[vector, Iterable]=(0, 0, 0),
                 size: float=0.2,
                 col: color=color.white):
        super().__init__(
            simul=simul,
            obj1=obj1,
            obj2=obj2
        )
        self._pos = convert_to_tensor(pos)
        self._pos_from_obj1 = self._pos - obj1.pos
        self._pos_from_obj2 = self._pos - obj2.pos

        size_vec = vector(size, size, size)
        self._ball = sphere(
            pos=convert_to_vector(self._pos),
            color=col,
            size=size_vec * 2
        )
        self._arm1 = cylinder(
            pos=convert_to_vector(self._pos),
            axis=convert_to_vector(obj1.pos - self._pos),
            radius=size,
            color=obj1.color
        )
        self._arm2 = cylinder(
            pos=convert_to_vector(self._pos),
            axis=convert_to_vector(obj2.pos - self._pos),
            radius=size,
            color=obj2.color
        )

    @property
    def g(self) -> torch.Tensor:
        return \
            self._obj1.to_global(self._pos_from_obj1) \
          - self._obj2.to_global(self._pos_from_obj2)

    @property
    def G(self) -> torch.Tensor:
        res = torch.zeros((3, 6 * self._simul.num_objects))

        idx1 = self._obj1.index
        idx2 = self._obj2.index

        pos_from_obj1 = self._pos - self._obj1.pos
        pos_from_obj2 = self._pos - self._obj2.pos

        # translation constraints
        res[0:3, 6 * idx1: 6 * idx1 + 6] = torch.concatenate([
            self._obj1.lin_vel_coeff_mat,
            -skew_matrix(pos_from_obj1) @ self._obj1.ang_vel_coeff_mat
        ], axis=1)

        res[0:3, 6 * idx2: 6 * idx2 + 6] = torch.concatenate([
            -self._obj2.lin_vel_coeff_mat,
            skew_matrix(pos_from_obj2) @ self._obj2.ang_vel_coeff_mat
        ], axis=1)

        return res
    
    def update(self):
        self._pos = self._obj1.to_global(self._pos_from_obj1)
        self._ball.pos = convert_to_vector(self._pos)

        self._arm1.pos = convert_to_vector(self._pos)
        self._arm1.axis = convert_to_vector(self._obj1.pos - self._pos)

        self._arm2.pos = convert_to_vector(self._pos)
        self._arm2.axis = convert_to_vector(self._obj2.pos - self._pos)
    

