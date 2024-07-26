import torch
from typing import Union, Iterable
from vpython import *

from objects import BaseObject
from joints import BaseJoint
from utils import convert_to_vector, skew_matrix

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
        ball_pos = convert_to_vector(pos)
        self._pos_from_obj1 = torch.as_tensor((ball_pos - obj1.pos).value)
        self._pos_from_obj2 = torch.as_tensor((ball_pos - obj2.pos).value)

        size_vec = vector(size, size, size)
        self._ball = sphere(pos=ball_pos, color=col, size=size_vec * 2)
        self._arm1 = cylinder(
            pos=ball_pos,
            axis=obj1.pos - ball_pos,
            radius=size,
            color=obj1.color
        )
        self._arm2 = cylinder(
            pos=ball_pos,
            axis=obj2.pos - ball_pos,
            radius=size,
            color=obj2.color
        )

    @property
    def g(self) -> torch.Tensor:
        return \
            self._obj1.pos_tensor + self._obj1.rot_mat @ self._pos_from_obj1 \
          - self._obj2.pos_tensor - self._obj2.rot_mat @ self._pos_from_obj2

    @property
    def G(self) -> torch.Tensor:
        res = torch.zeros((3, 6 * self._simul.num_objects))

        idx1 = self._obj1.index
        idx2 = self._obj2.index

        pos_from_obj1 = torch.as_tensor(self._ball.pos.value) - self._obj1.pos_tensor
        pos_from_obj2 = torch.as_tensor(self._ball.pos.value) - self._obj2.pos_tensor

        # translation constraints
        # res[:, 6 * idx1: 6 * idx1 + 6] = torch.concatenate([
        #     self._obj1.lin_vel_coeff_mat,
        #     -skew_matrix(self._obj1.rot_mat @ self._pos_from_obj1)
        # ], axis=1)

        # res[:, 6 * idx2: 6 * idx2 + 6] = torch.concatenate([
        #     -self._obj2.lin_vel_coeff_mat,
        #     skew_matrix(self._obj2.rot_mat @ self._pos_from_obj2)
        # ], axis=1)

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
        pos_next = convert_to_vector(
            self._obj1.pos_tensor + self._obj1.rot_mat @ self._pos_from_obj1
        )
        self._ball.pos = pos_next

        self._arm1.pos = pos_next
        self._arm1.axis = self._obj1.pos - pos_next

        self._arm2.pos = pos_next
        self._arm2.axis = self._obj2.pos - pos_next
    

