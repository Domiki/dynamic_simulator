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
                 size: float=0.5,
                 col: color=color.white):
        super().__init__(
            simul=simul,
            obj1=obj1,
            obj2=obj2
        )
        hinge_pos = convert_to_vector(pos)
        self._pos_from_obj1 = torch.as_tensor((hinge_pos - obj1.pos).value)
        self._pos_from_obj2 = torch.as_tensor((hinge_pos - obj2.pos).value)

        self._axis = convert_to_vector(axis) * size
        self._axis_from_obj1 = \
            obj1.rot_mat.T @ torch.as_tensor(self._axis.value)
        self._axis_from_obj2 = \
            obj2.rot_mat.T @ torch.as_tensor(self._axis.value)

        self._hinge = cylinder(
            pos=hinge_pos - self._axis / 2,
            axis=self._axis,
            radius=size / 10,
            color=col
        )
        self._arm1 = box(
            pos=(hinge_pos + obj1.pos) / 2,
            axis=obj1.pos - hinge_pos,
            width=size,
            height=size / 5,
            color=obj1.color
        )
        self._arm2 = box(
            pos=(hinge_pos + obj2.pos) / 2,
            axis=obj2.pos - hinge_pos,
            width=size,
            height=size / 5,
            color=obj2.color
        )

    @property
    def g(self) -> torch.Tensor:
        g1 = \
            (self._obj1.pos_tensor + self._obj1.rot_mat @ self._pos_from_obj1) \
          - (self._obj2.pos_tensor + self._obj2.rot_mat @ self._pos_from_obj2)
        
        g2 = \
            self._obj1.rot_mat @ self._axis_from_obj1 \
          - self._obj2.rot_mat @ self._axis_from_obj2
        
        return torch.hstack([g1, g2])
    
    @property
    def G(self) -> torch.Tensor:
        res = torch.zeros((6, 6 * self._simul.num_objects))

        idx1 = self._obj1.index
        idx2 = self._obj2.index

        # translation constraints
        pos_from_obj1 = self._hinge.pos - self._obj1.pos
        pos_from_obj2 = self._hinge.pos - self._obj2.pos

        res[0:3, 6 * idx1: 6 * idx1 + 6] = torch.concatenate([
            self._obj1.lin_vel_coeff_mat,
            -skew_matrix(pos_from_obj1) @ self._obj1.ang_vel_coeff_mat
        ], axis=1)

        res[0:3, 6 * idx2: 6 * idx2 + 6] = torch.concatenate([
            -self._obj2.lin_vel_coeff_mat,
            skew_matrix(pos_from_obj2) @ self._obj2.ang_vel_coeff_mat
        ], axis=1)

        # rotation constraints
        obj1_axis_skew = skew_matrix(self._obj1.rot_mat @ self._axis_from_obj1)
        obj2_axis_skew = skew_matrix(self._obj2.rot_mat @ self._axis_from_obj2)
        
        res[3:6, 6 * idx1: 6 * idx1 + 6] = torch.concatenate([
            torch.zeros((3, 3)),
            obj1_axis_skew @ self._obj1.ang_vel_coeff_mat
        ], axis=1)

        res[3:6, 6 * idx2: 6 * idx2 + 6] = torch.concatenate([
            torch.zeros((3, 3)),
            -obj2_axis_skew @ self._obj2.ang_vel_coeff_mat
        ], axis=1)

        return res
    
    def update(self):
        pos_next = convert_to_vector(
            self._obj1.pos_tensor + self._obj1.rot_mat @ self._pos_from_obj1
        )
        self._hinge.pos = pos_next - self._axis / 2
        self._hinge.axis = convert_to_vector(
            self._obj1.rot_mat @ torch.as_tensor(self._axis.value)
        )

        self._arm1.pos = (pos_next + self._obj1.pos) / 2
        self._arm1.axis = self._obj1.pos - pos_next

        self._arm2.pos = (pos_next + self._obj2.pos) / 2
        self._arm2.axis = self._obj2.pos - pos_next