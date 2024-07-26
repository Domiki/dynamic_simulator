import torch
from typing import Union, Iterable
from vpython import *

from objects import BaseObject
from joints import BaseJoint
from utils import convert_to_vector, skew_matrix

class HingeJoint(BaseJoint):
    def __init__(self,
                 simul,
                 obj1: BaseObject,
                 obj2: BaseObject,
                 pos: Union[vector, Iterable]=(0, 0, 0),
                 axis: Union[vector, Iterable]=(1, 0, 0),
                 size: float=0.2,
                 col: color=color.white):
        super().__init__(
            simul=simul,
            obj1=obj1,
            obj2=obj2
        )
        hinge_pos = convert_to_vector(pos)
        self._pos_from_obj1 = torch.as_tensor((hinge_pos - obj1.pos).value)
        self._pos_from_obj2 = torch.as_tensor((hinge_pos - obj2.pos).value)

        self._axis = convert_to_vector(axis)

        # size_vec = vector(size, size / 10, size)
        self._hinge = cylinder(
            pos=hinge_pos,
            axis=self._axis,
            radius=size,
            color=col
        )

    @property
    def g(self) -> torch.Tensor:
        g1 = \
            (self._obj1.pos_tensor + self._obj1.rot_mat @ self._pos_from_obj1) \
          - (self._obj2.pos_tensor + self._obj2.rot_mat @ self._pos_from_obj2)
        
        g2 = torch.linalg.cross(
            self._obj1.rot_mat @ torch.as_tensor(self._axis.value),
            self._obj2.rot_mat @ torch.as_tensor(self._axis.value)
        )

        return torch.hstack([g1, g2])
    
    @property
    def G(self) -> torch.Tensor:
        res = torch.zeros((6, 6 * self._simul.num_objects))

        idx1 = self._obj1.index
        idx2 = self._obj2.index

        pos_from_obj1 = self._hinge.pos - self._obj1.pos
        pos_from_obj2 = self._hinge.pos - self._obj2.pos

        obj1_ang_vel = skew_matrix(pos_from_obj1) @ self._obj1.ang_vel_coeff_mat
        obj2_ang_vel = skew_matrix(pos_from_obj2) @ self._obj2.ang_vel_coeff_mat

        # translation constraints
        res[0:3, 6 * idx1: 6 * idx1 + 6] = torch.concatenate([
            self._obj1.lin_vel_coeff_mat,
            -obj1_ang_vel
        ], axis=1)

        res[0:3, 6 * idx2: 6 * idx2 + 6] = torch.concatenate([
            -self._obj2.lin_vel_coeff_mat,
            obj2_ang_vel
        ], axis=1)

        # rotation constraints
        res[3:6, 6 * idx1: 6 * idx1 + 6] = torch.concatenate([
            torch.zeros((3, 3)),
            -skew_matrix(self._axis) @ obj1_ang_vel
        ], axis=1)

        res[3:6, 6 * idx2: 6 * idx2 + 6] = torch.concatenate([
            torch.zeros((3, 3)),
            skew_matrix(self._axis) @ obj2_ang_vel
        ], axis=1)

        return res
    
    def update(self):
        pos_next = convert_to_vector(
            torch.as_tensor(self._obj1.pos.value)
          + self._obj1.rot_mat @ torch.as_tensor(self._pos_from_obj1.value)
        )
        self._hinge.pos = pos_next
