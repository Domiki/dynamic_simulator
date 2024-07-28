import torch
from vpython import rate

from objects import BaseObject
from joints import BaseJoint

class Simulation:
    def __init__(self, fps: int, pause: bool=False):
        self._fps = fps
        self._running = True
        self._pause = pause
        self._object_list: list[BaseObject] = []
        self._joint_list: list[BaseJoint] = []

        self._M = None
        self._epsilon_mat = None
        
        self._tau = 0.005
        self._Lambda = 1 / (1 + 4 * self._tau / self.h)
        self._lambda = None
        self._epsilon = self._Lambda * 4 / (self.h ** 2) * 1e-8
    
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
    
    @property
    def num_objects(self) -> int:
        return len(self._object_list)

    @property
    def num_joints(self) -> int:
        return len(self._joint_list)

    @running.setter
    def running(self, value: bool) -> None:
        self._running = value

    @pause.setter
    def pause(self, value: bool) -> None:
        self._pause = value

    def init(self) -> None:
        self._M = torch.vstack([
            torch.hstack([
                torch.zeros((6, 6 * i)),
                obj.mass * torch.eye(6),
                torch.zeros((6, 6 * (self.num_objects - i - 1)))
            ]) for i, obj in enumerate(self._object_list)
        ])

        g = torch.hstack([
            joint.g for joint in self._joint_list
        ])
        self._epsilon_mat = torch.ones((len(g), len(g))) * self._epsilon
        self._lambda = torch.zeros((len(g),))

    def add_object(self, object: BaseObject) -> int:
        self._object_list.append(object)
        return len(self._object_list) - 1
    
    def add_joint(self, joint: BaseJoint) -> int:
        self._joint_list.append(joint)
        return len(self._joint_list) - 1

    def update(self):
        v = torch.hstack([
            obj.v for obj in self._object_list
        ]).reshape(-1, 1)

        g = torch.hstack([
            joint.g for joint in self._joint_list
        ]).reshape(-1, 1)
        
        G = torch.vstack([
            joint.G for joint in self._joint_list
        ])

        V_q = torch.zeros((6 * self.num_objects, 1))
        for i, obj in enumerate(self._object_list):
            V_q[6 * i + 1][0] = (0 if obj.pos_fixed else obj.mass) * 9.80665

        L_mat = torch.vstack([
            torch.hstack([self._M, -G.T]),
            torch.hstack([G, self._epsilon_mat])
        ])

        L_mat_nonzero_cols = torch.any(L_mat > self._epsilon, dim=0).nonzero()

        sub_L_mat = L_mat[L_mat_nonzero_cols, L_mat_nonzero_cols.reshape(-1,)]
        sub_L_mat_inv = torch.linalg.inv(sub_L_mat)

        L_mat_inv = torch.zeros_like(L_mat)
        L_mat_inv[L_mat_nonzero_cols, L_mat_nonzero_cols.reshape(-1,)] = \
            sub_L_mat_inv

        R_mat = torch.vstack([
            self._M @ v - self.h * V_q,
            -4 * self._Lambda / self.h * g + self._Lambda * G @ v
        ])

        result = torch.squeeze(L_mat_inv @ R_mat)
        for i, obj in enumerate(self._object_list):
            v_next = result[i * 6:(i + 1) * 6]
            q_next = obj.q + self.h * v_next
            obj.update(q_next)
        self._lambda = result[6 * self.num_objects:]

        for joint in self._joint_list:
            joint.update()

    def run(self):
        self.init()
        while self.running:
            if not self.pause:
                self.update()
                print(torch.norm(self._joint_list[0]._obj2.pos - self._joint_list[0]._obj1.pos))
            rate(self.fps)