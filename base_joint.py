import numpy as np
from vpython import *
from base_object import BaseObject

class BaseJoint:
    def __init__(self,
                 obj1: BaseObject,
                 obj2: BaseObject):
        self._index = -1
        self._obj1 = obj1
        self._obj2 = obj2
    
    @property
    def index(self) -> int:
        return self._index

    @index.setter
    def index(self, value: int) -> None:
        self._index = value
    
    def g(self) -> np.ndarray:
        pass
    
    def G(self, obj_len: int) -> np.ndarray:
        pass

    def update(self) -> None:
        pass