import numpy as np

from base_object import BaseObject
from base_joint import BaseJoint

class Simulation:
    def __init__(self, fps: int):
        self._fps = fps
        self._running = True
        self._pause = False
        self._object_list: list[BaseObject] = []
        self._joint_list: list[BaseJoint] = []
    
    @property
    def fps(self) -> int:
        return self._fps
    
    @property
    def h(self) -> float:
        return 1 / self._fps
    
    @property
    def running(self) -> bool:
        return self._running
    
    @property
    def pause(self) -> bool:
        return self._pause

    @running.setter
    def running(self, value: bool) -> None:
        self._running = value

    @pause.setter
    def pause(self, value: bool) -> None:
        self._pause = value

    def add_object(self, object: BaseObject) -> None:
        object.index = len(self._object_list)
        self._object_list.append(object)
    
    def add_joint(self, joint: BaseJoint) -> None:
        joint.index = len(self._joint_list)
        self._joint_list.append(joint)

    def update(self):
        obj_len = len(self._object_list)

        q = np.hstack([
            obj.q for obj in self._object_list
        ]).reshape(-1, 1)
        q_before = np.hstack([
            obj.q_before for obj in self._object_list
        ]).reshape(-1, 1)

        M = np.vstack([
            np.hstack([
                np.zeros((6, 6 * i), dtype=np.float32),
                obj.mass * np.eye(6),
                np.zeros((6, 6 * (obj_len - i - 1)), dtype=np.float32)
            ]) for i, obj in enumerate(self._object_list)
        ])
        
        G = np.vstack([
            joint.G(obj_len) for joint in self._joint_list
        ])

        L_mat = np.vstack([
            np.hstack([M, -G.T]),
            np.hstack([G, np.zeros((len(G), len(G)), dtype=np.float32)])
        ])

        V_q = np.zeros((6 * obj_len, 1), dtype=np.float32)
        for i, obj in enumerate(self._object_list):
            V_q[6 * i + 1][0] = (0 if obj.pos_fixed else obj.mass) * 9.81
        
        R_mat = np.vstack([
            M @ (2 * q - q_before) - (self.h ** 2) * V_q,
            G @ q
        ])

        result = np.linalg.inv(L_mat) @ R_mat
        for i, obj in enumerate(self._object_list):
            q_next = result[i * 6:(i + 1) * 6]
            obj.update(q_next)
        
        for joint in self._joint_list:
            joint.update()
