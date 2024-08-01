import torch
from typing import TYPE_CHECKING
from vpython import *

if TYPE_CHECKING:
    from simulation import Simulation
    from objects import BaseObject

class BaseJoint:
    def __init__(self,
                 simul: 'Simulation',
                 obj1: 'BaseObject',
                 obj2: 'BaseObject'):
        self._simul = simul
        self._index = simul.add_joint(self)
        self._obj1 = obj1
        self._obj2 = obj2

    @property
    def g(self) -> torch.Tensor:
        pass
    
    @property
    def G(self) -> torch.Tensor:
        pass

    def update(self) -> None:
        pass