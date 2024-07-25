import numpy as np

from vpython import *
from base_joint import BaseJoint
from base_object import BaseObject
from utils import skew_matrix

class BallJoint(BaseJoint):
    def __init__(self,
                 obj1: BaseObject,
                 obj2: BaseObject,
                 pos: tuple[float],
                 size: float=1,
                 col: color=color.black):
        super().__init__(
            obj1=obj1,
            obj2=obj2
        )
        hinge_pos = vector(*pos)
        self._pos_from_obj1 = hinge_pos - obj1.pos
        self._pos_from_obj2 = hinge_pos - obj2.pos

        size_vec = vector(size, size, size)
        self._hinge = sphere(pos=hinge_pos, color=col, size=size_vec)
        self._arm1 = cylinder(
            pos=hinge_pos,
            axis=obj1.pos - hinge_pos,
            radius=size / 5,
            color=obj1.color
        )
        self._arm2 = cylinder(
            pos=hinge_pos,
            axis=obj2.pos - hinge_pos,
            radius=size / 5,
            color=obj2.color
        )

    def G(self, obj_len: int) -> np.ndarray:
        res = np.zeros((3, 6 * obj_len))

        idx1 = self._obj1.index
        idx2 = self._obj2.index

        pos_from_obj1 = self._hinge.pos - self._obj1.pos
        pos_from_obj2 = self._hinge.pos - self._obj2.pos

        res[:, 6 * idx1: 6 * idx1 + 6] = np.concatenate([
            (not self._obj1.pos_fixed) * np.eye(3),
            -skew_matrix(pos_from_obj1) @ self._obj1.ang_vel_coeff_mat
        ], axis=1)

        res[:, 6 * idx2: 6 * idx2 + 6] = np.concatenate([
            -(not self._obj2.pos_fixed) * np.eye(3),
            skew_matrix(pos_from_obj2) @ self._obj2.ang_vel_coeff_mat
        ], axis=1)

        return res
    
    def update(self):
        pos_next = vector(*(
            np.array(self._obj1.pos.value)
          + self._obj1.rot_mat @ np.array(self._pos_from_obj1.value)
        ))
        self._hinge.pos = pos_next

        self._arm1.pos = pos_next
        self._arm1.axis = self._obj1.pos - pos_next

        self._arm2.pos = pos_next
        self._arm2.axis = self._obj2.pos - pos_next
    

