from vpython import *
from typing import Union, Iterable

from objects import BaseObject
from utils import convert_to_vector

class Box(BaseObject):
    def __init__(self,
                 simul,
                 pos: Union[vector, Iterable]=(0, 0, 0),
                 dir: Union[vector, Iterable]=(0, 0, 0),
                 mass: float=1,
                 col: color=color.white,
                 pos_fixed: bool=False,
                 rot_fixed: bool=False):
        super().__init__(
            simul=simul,
            pos=pos,
            mass=mass,
            col=col,
            dir=dir,
            pos_fixed=pos_fixed,
            rot_fixed=rot_fixed
        )

        size = vector(mass, mass, mass)
        self._obj = box(
            pos=convert_to_vector(self._pos),
            color=col,
            size=size
        )
