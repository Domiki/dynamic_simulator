import torch
from typing import Union, Iterable, TYPE_CHECKING
from vpython import *

from joints import BaseJoint
from utils import *

if TYPE_CHECKING:
    from simulation import Simulation
    from objects import BaseObject

class UniversalJoint(BaseJoint):
    def __init__(self,
                 simul: 'Simulation',
                 obj1: 'BaseObject',
                 obj2: 'BaseObject',
                 pos: Union[vector, Iterable]=(0, 0, 0),
                 axis: Union[vector, Iterable]=(1, 0, 0),
                 size: float=0.8,
                 col: color=color.white):
        super().__init__(
            simul=simul,
            obj1=obj1,
            obj2=obj2
        )
        self._pos = convert_to_tensor(pos)
        self._pos_from_obj1 = self._pos - obj1.pos
        self._pos_from_obj2 = self._pos - obj2.pos

        # get universal joint's axises
        axis = convert_to_tensor(axis)

        temp_axis = torch.linalg.cross(self._pos_from_obj1, axis)
        self._axis1 = torch.linalg.cross(self._pos_from_obj1, temp_axis)
        self._axis2 = torch.linalg.cross(self._pos_from_obj2, self._axis1)
        self._axis1 *= size / torch.norm(self._axis1)
        self._axis2 *= size / torch.norm(self._axis2)
        self._axis1_from_obj1 = obj1.rotate_to_local(self._axis1)
        self._axis2_from_obj2 = obj2.rotate_to_local(self._axis2)

        axes = torch.cat((self._axis1, self._axis2))
        assert not torch.any(torch.isnan(axes)), "Axis error"

        self._cylinder1 = cylinder(
            pos=convert_to_vector(self._pos - self._axis1 / 2),
            axis=convert_to_vector(self._axis1),
            radius = size / 10,
            color=col
        )
        self._cylinder2 = cylinder(
            pos=convert_to_vector(self._pos - self._axis2 / 2),
            axis=convert_to_vector(self._axis2),
            radius = size / 10,
            color=col
        )

        width = size / 10
        height = size / 4
        self._arm1_1 = box(
            pos=convert_to_vector((self._pos + obj1.pos + self._axis1) / 2),
            axis=convert_to_vector(obj1.pos - self._pos),
            width=width,
            height=height,
            color=obj1.color
        )
        self._arm1_2 = box(
            pos=convert_to_vector((self._pos + obj1.pos - self._axis1) / 2),
            axis=convert_to_vector(obj1.pos - self._pos),
            width=width,
            height=height,
            color=obj1.color
        )
        self._arm2_1 = box(
            pos=convert_to_vector((self._pos + obj2.pos + self._axis2) / 2),
            axis=convert_to_vector(obj2.pos - self._pos),
            width=width,
            height=height,
            color=obj2.color
        )
        self._arm2_2 = box(
            pos=convert_to_vector((self._pos + obj2.pos - self._axis2) / 2),
            axis=convert_to_vector(obj2.pos - self._pos),
            width=width,
            height=height,
            color=obj2.color
        )

    @property
    def g(self) -> torch.Tensor:
        g1 = \
            self._obj1.to_global(self._pos_from_obj1) \
          - self._obj2.to_global(self._pos_from_obj2)
        
        g2 = torch.dot(
            self._obj1.rotate_to_global(self._axis1_from_obj1),
            self._obj2.rotate_to_global(self._axis2_from_obj2)
        )
        
        return torch.hstack([g1, g2])

    @property
    def G(self) -> torch.Tensor:
        res = torch.zeros((4, 6 * self._simul.num_objects))

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

        # rotation constraints
        axis1 = self._obj1.rotate_to_global(self._axis1_from_obj1)
        axis2 = self._obj2.rotate_to_global(self._axis2_from_obj2)

        res[3, 6 * idx1: 6 * idx1 + 6] = torch.concatenate([
            torch.zeros((3,)),
            -axis2 @ skew_matrix(axis1)
        ])

        res[3, 6 * idx2: 6 * idx2 + 6] = torch.concatenate([
            torch.zeros((3,)),
            -axis1 @ skew_matrix(axis2)
        ])

        return res
    
    def update(self):
        self._pos = self._obj1.to_global(self._pos_from_obj1)
        self._axis1 = self._obj1.rotate_to_global(self._axis1_from_obj1)
        self._axis2 = self._obj2.rotate_to_global(self._axis2_from_obj2)

        self._cylinder1.pos = convert_to_vector(self._pos - self._axis1 / 2)
        self._cylinder2.pos = convert_to_vector(self._pos - self._axis2 / 2)
        self._cylinder1.axis = convert_to_vector(self._axis1)
        self._cylinder2.axis = convert_to_vector(self._axis2)

        self._arm1_1.pos = \
            convert_to_vector((self._pos + self._obj1.pos + self._axis1) / 2)
        self._arm1_2.pos = \
            convert_to_vector((self._pos + self._obj1.pos - self._axis1) / 2)
        self._arm2_1.pos = \
            convert_to_vector((self._pos + self._obj2.pos + self._axis2) / 2)
        self._arm2_2.pos = \
            convert_to_vector((self._pos + self._obj2.pos - self._axis2) / 2)
        
        self._arm1_1.axis = convert_to_vector(self._obj1.pos - self._pos)
        self._arm1_2.axis = convert_to_vector(self._obj1.pos - self._pos)
        self._arm2_1.axis = convert_to_vector(self._obj2.pos - self._pos)
        self._arm2_2.axis = convert_to_vector(self._obj2.pos - self._pos)
    