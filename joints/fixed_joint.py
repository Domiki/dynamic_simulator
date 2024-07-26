import torch
from vpython import *

from objects import BaseObject
from joints import BaseJoint
from utils import skew_matrix

class FixedJoint(BaseJoint):
    def __init__(self,
                 simul,
                 obj1: BaseObject,
                 obj2: BaseObject,
                 size: float=0.2,
                 col: color=color.white):
        super().__init__(
            simul=simul,
            obj1=obj1,
            obj2=obj2
        )
        mid_pos = (obj1.pos + obj2.pos) / 2
        self._pos_from_obj1 = torch.as_tensor((mid_pos - obj1.pos).value)
        self._pos_from_obj2 = torch.as_tensor((mid_pos - obj2.pos).value)
        self._obj1_init_dir = obj1.dir_tensor
        self._obj2_init_dir = obj2.dir_tensor

        self._arm = cylinder(
            pos=obj1.pos,
            axis=obj2.pos - obj1.pos,
            radius=size,
            color=col
        )

    @property
    def g(self) -> torch.Tensor:
        g1 = \
            (self._obj1.pos_tensor + self._obj1.rot_mat @ self._pos_from_obj1) \
          - (self._obj2.pos_tensor + self._obj2.rot_mat @ self._pos_from_obj2)

        g2 = \
            (self._obj1.dir_tensor - self._obj1_init_dir) \
          - (self._obj2.dir_tensor - self._obj2_init_dir)
        
        return torch.hstack([g1, g2])

    @property
    def G(self) -> torch.Tensor:
        res = torch.zeros((6, 6 * self._simul.num_objects))

        idx1 = self._obj1.index
        idx2 = self._obj2.index

        mid_pos = (self._obj1.pos_tensor + self._obj2.pos_tensor) / 2
        pos_from_obj1 = mid_pos - self._obj1.pos_tensor
        pos_from_obj2 = mid_pos - self._obj2.pos_tensor

        # translation constraints
        # res[0:3, 6 * idx1: 6 * idx1 + 6] = torch.concatenate([
        #     self._obj1.lin_vel_coeff_mat,
        #     -skew_matrix(self._obj1.rot_mat @ self._pos_from_obj1)
        # ], axis=1)

        # res[0:3, 6 * idx2: 6 * idx2 + 6] = torch.concatenate([
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

        # rotation constraints
        res[3:6, 6 * idx1: 6 * idx1 + 6] = torch.concatenate([
            torch.zeros((3, 3)),
            torch.eye(3)
        ], axis=1)

        res[3:6, 6 * idx2: 6 * idx2 + 6] = torch.concatenate([
            torch.zeros((3, 3)),
            -torch.eye(3)
        ], axis=1)

        return res
    
    def update(self):
        self._arm.pos = self._obj1.pos
        self._arm.axis = self._obj2.pos - self._obj1.pos