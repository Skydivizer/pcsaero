# -*- coding: utf-8 -*-
"""This module defines some utility functions to transform points in 2d
space."""

import numpy as np


def make_homogenous(x):
    """Makes a matrix of coordinates as row vectors homogenous.

    Arguments:
        x (float[w, h]): Matrix of row vectors.

    Returns:
        float[w + 1, h]: Same as x with added column vector of ones.
    """
    return np.vstack([x, np.ones(x.shape[1], dtype=x.dtype)])


def unmake_homogenous(x):
    """'Unmakes' a matrix of homogenous coordinats as row vectors homogenous.

    Arguments:
        x (float[w, h]): Matrix of row vectors.

    Returns:
        float[w - 1, h]: x normalized by last column vector, without the last 
            column vector.
    """
    return x[:-1] / x[-1]


def R(theta):
    """Create a 2d rotation matrix for homogenous coordinates
    
    Arguments:
        theta (float): Angle to rotate around counter clockwise.

    Returns:
        float[3, 3]: Rotation matrix.
    """
    cos = np.cos(theta)
    sin = np.sin(theta)

    return np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]])


def T(c):
    """Create a 2d translation matrix for homogenous coordinates
    
    Arguments:
        c (float[2]): Translation vector in 2D space.

    "Returns":
        float[3, 3]: Transformation matrix.
    """
    x, y = c

    return np.array([[1, 0, x], [0, 1, y], [0, 0, 1]])


def rotate(x, theta, c):
    """Rotate a matrix of 2d coordinates as row vectors theta degrees around a
    point c

    Arguments:
        x (float[w, h]): Matrix of row vectors.
        theta (float): Angle to rotate around counter clockwise.
        c (float[2]): Rotation point in 2D space.

    Returns:
        float[w, h]: Points of x rotated theta degrees around c counter
            clockwise.
    """
    x = make_homogenous(x)

    c = np.array(c)

    M = T(c) @ R(theta) @ T(-c)

    x = M @ x

    return unmake_homogenous(x)


