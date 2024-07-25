import numpy as np
from vpython import *

class BaseObject:
    def __init__(self,
                 pos: tuple[float],
                 mass: float=1,
                 col: color=color.black,
                 dir: tuple[float]=(0, 0, 0),
                 pos_fixed: bool=False,
                 rot_fixed: bool=False):
        self._pos = vector(*pos)
        self._pos_before = vector(*pos)
        self._dir = vector(*dir)
        self._dir_before = vector(*dir)
        self._mass = mass
        self._color = col
        self._pos_fixed = pos_fixed
        self._rot_fixed = rot_fixed
        self._index = -1
        self._obj: standardAttributes = None
    
    @property
    def pos(self) -> vector:
        return self._pos
    
    @property
    def dir(self) -> vector:
        return self._dir
    
    @property
    def pos_fixed(self) -> bool:
        return self._pos_fixed

    @property
    def rot_fixed(self) -> bool:
        return self._rot_fixed
    
    @property
    def pos_before(self) -> vector:
        return self._pos_before
    
    @property
    def dir_before(self) -> vector:
        return self._dir_before

    @property
    def mass(self) -> float:
        return self._mass
    
    @property
    def color(self) -> color:
        return self._color
    
    @property
    def index(self) -> int:
        return self._index

    @property
    def obj(self) -> standardAttributes:
        return self._obj
    
    @property
    def q(self) -> np.ndarray:
        return np.array(self._pos.value + self._dir.value)
    
    @property
    def q_before(self) -> np.ndarray:
        return np.array(self._pos_before.value + self._dir_before.value)

    @index.setter
    def index(self, value: int) -> None:
        self._index = value

    @pos.setter
    def pos(self, value: vector) -> None:
        self._pos_before = vector(self._pos)
        self._pos = vector(value)
        self._obj.pos = self._pos
    
    @dir.setter
    def dir(self, value: vector) -> None:
        self._dir_before = vector(self._dir)
        self._dir = vector(value)
        self._obj.up = vector(*self.rot_mat[:, 1])
        self._obj.axis = vector(*self.rot_mat[:, 0])
    
    @property
    def ang_vel_coeff_mat(self) -> np.ndarray:
        # phi: x, theta: y, ksi: z
        x, y, z = self._dir.value
        return np.array([
            [0, cos(x),  sin(y) * sin(x)],
            [0, sin(x), -sin(y) * cos(x)],
            [1,      0,           cos(y)],
        ])
    
    @property
    def rot_mat(self) -> np.ndarray:
        # phi: x, theta: y, psi: z
        x, y, z = self._dir.value
        return np.array([
            [
                 cos(z) * cos(x) - cos(y) * sin(x) * sin(z),
                -sin(z) * cos(x) - cos(y) * sin(x) * cos(z),
                 sin(y) * sin(x)
            ],
            [
                 cos(z) * sin(x) + cos(y) * cos(x) * sin(z),
                -sin(z) * sin(x) + cos(y) * cos(x) * cos(z),
                -sin(y) * cos(x)
            ],
            [
                sin(y) * sin(z),
                sin(y) * cos(z),
                cos(y)
            ],
        ])

    def update(self, q: np.ndarray) -> None:
        _pos = q[0:3]
        _dir = q[3:6]

        self.pos = vector(*_pos)
        self.dir = vector(*_dir)