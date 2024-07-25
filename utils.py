import numpy as np
from vpython import vector

def skew_matrix(vec: vector) -> np.ndarray:
    return np.array([
        [0, -vec.z, vec.y],
        [vec.z, 0, -vec.x],
        [-vec.y, vec.x, 0],
    ])