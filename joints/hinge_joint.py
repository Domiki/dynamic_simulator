import torch
from typing import Union, Iterable
from vpython import *

from objects import BaseObject
from joints import BaseJoint
from utils import *

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
        self._pos = convert_to_tensor(pos)
        self._pos_from_obj1 = self._pos - obj1.pos
        self._pos_from_obj2 = self._pos - obj2.pos

        self._axis = convert_to_tensor(axis)
        self._axis *= size / torch.norm(self._axis)
        self._axis_from_obj1 = obj1.rotate_to_local(self._axis)
        self._axis_from_obj2 = obj2.rotate_to_local(self._axis)

        self._hinge = cylinder(
            pos=convert_to_vector(self._pos - self._axis / 2),
            axis=convert_to_vector(self._axis),
            radius=size / 10,
            color=col
        )
        self._arm1 = box(
            pos=convert_to_vector((self._pos + obj1.pos) / 2),
            axis=convert_to_vector(obj1.pos - self._pos),
            width=size / 5,
            height=size,
            color=obj1.color
        )
        self._arm2 = box(
            pos=convert_to_vector((self._pos + obj2.pos) / 2),
            axis=convert_to_vector(obj2.pos - self._pos),
            width=size / 5,
            height=size,
            color=obj2.color
        )

    @property
    def g(self) -> torch.Tensor:
        g1 = \
            self._obj1.to_global(self._pos_from_obj1) \
          - self._obj2.to_global(self._pos_from_obj2)
        
        g2 = torch.linalg.cross(
            self._obj1.rotate_to_global(self._axis_from_obj1),
            self._obj2.rotate_to_global(self._axis_from_obj2)
        )
        
        return torch.hstack([g1, g2])
    
    @property
    def G(self) -> torch.Tensor:
        res = torch.zeros((6, 6 * self._simul.num_objects))

        idx1 = self._obj1.index
        idx2 = self._obj2.index

        # translation constraints
        pos_from_obj1 = self._pos - self._obj1.pos
        pos_from_obj2 = self._pos - self._obj2.pos

        res[0:3, 6 * idx1: 6 * idx1 + 6] = torch.concatenate([
            self._obj1.lin_vel_coeff_mat,
            -skew_matrix(pos_from_obj1) @ self._obj1.ang_vel_coeff_mat
        ], axis=1)

        res[0:3, 6 * idx2: 6 * idx2 + 6] = torch.concatenate([
            -self._obj2.lin_vel_coeff_mat,
            skew_matrix(pos_from_obj2) @ self._obj2.ang_vel_coeff_mat
        ], axis=1)

        # rotation constraints
        obj1_axis_skew = skew_matrix(self._obj1.to_global(self._axis_from_obj1))
        obj2_axis_skew = skew_matrix(self._obj2.to_global(self._axis_from_obj2))
        
        res[3:6, 6 * idx1: 6 * idx1 + 6] = torch.concatenate([
            torch.zeros((3, 3)),
            obj2_axis_skew @ obj1_axis_skew @ self._obj1.ang_vel_coeff_mat
        ], axis=1)

        res[3:6, 6 * idx2: 6 * idx2 + 6] = torch.concatenate([
            torch.zeros((3, 3)),
            -obj1_axis_skew @ obj2_axis_skew @ self._obj2.ang_vel_coeff_mat
        ], axis=1)

        return res
    
    def update(self):
        self._pos = self._obj1.to_global(self._pos_from_obj1)
        self._axis = self._obj1.rotate_to_global(self._axis_from_obj1)

        self._hinge.pos = convert_to_vector(self._pos - self._axis / 2)
        self._hinge.axis = convert_to_vector(self._axis)

        self._arm1.pos = convert_to_vector((self._pos + self._obj1.pos) / 2)
        self._arm1.axis = convert_to_vector(self._obj1.pos - self._pos)
        self._arm1.up = convert_to_vector(self._axis)

        self._arm2.pos = convert_to_vector((self._pos + self._obj2.pos) / 2)
        self._arm2.axis = convert_to_vector(self._obj2.pos - self._pos)
        self._arm2.up = convert_to_vector(self._axis)