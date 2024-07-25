from base_object import BaseObject
from vpython import *

class Box(BaseObject):
    def __init__(self,
                 pos: tuple[float],
                 mass: float=1,
                 col: color=color.black,
                 dir: tuple[float]=(0, 0, 0),
                 pos_fixed: bool=False,
                 rot_fixed: bool=False):
        super().__init__(
            pos=pos,
            mass=mass,
            col=col,
            dir=dir,
            pos_fixed=pos_fixed,
            rot_fixed=rot_fixed
        )

        size = vector(mass, mass, mass)
        self._obj = box(pos=self._pos, color=col, size=size)
