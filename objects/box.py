from typing import Union, Iterable, TYPE_CHECKING
from vpython import *

from objects import BaseObject
from utils import *

if TYPE_CHECKING:
    from simulation import Simulation

class Box(BaseObject):
    def __init__(self,
                 simul: 'Simulation',
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

        self._obj = box(
            pos=convert_to_vector(self._pos),
            color=col,
            length=mass,
            height=mass,
            width=mass
        )
