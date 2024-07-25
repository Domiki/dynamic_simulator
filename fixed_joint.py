from vpython import *
from base_joint import BaseJoint
from base_object import BaseObject

class FixedJoint(BaseJoint):
    def __init__(self,
                 obj1: BaseObject,
                 obj2: BaseObject,
                 pos: tuple[float],
                 col: color=color.black):
        super().__init__(
            obj1=obj1,
            obj2=obj2
        )
        pos = vector(*pos)
        self._pos_from_obj1 = pos - obj1.pos
        self._pos_from_obj2 = pos - obj2.pos

        self._arm1 = cylinder()