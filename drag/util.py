import numpy as np

def make_homogenous(x):
    "Makes a matrix of coordinates as row vectors homogenous"
    return np.vstack([x, np.ones(x.shape[1], dtype=x.dtype)])

def unmake_homogenous(x):
    "'Unmakes' a matrix of homogenous coordinats as row vectors homogenous"
    return x[:-1] / x[-1]

def R(theta):
    "Returns a 2d rotation matrix for homogenous coordinates"
    cos = np.cos(theta)
    sin = np.sin(theta)

    return np.array([
        [cos, -sin, 0],
        [sin, cos, 0],
        [0, 0, 1]
    ])

def T(c):
    "Returns a 2d translation matrix for homogenous coordinates"
    x, y = c

    return np.array([
        [1, 0, x],
        [0, 1, y],
        [0, 0, 1]
    ])


def rotate(x, theta, c):
    """Rotate a matrix of 2d coordinates as row vectors theta degrees around a
    point c"""
    x = make_homogenous(x)

    c = np.array(c)

    M = T(c) @ R(theta) @ T(-c)

    x = M @ x

    return unmake_homogenous(x)
