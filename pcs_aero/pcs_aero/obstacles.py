# -*- coding: utf-8 -*-
"""Defines the obstacle creation.

The strings in the *_names attributes can be passed to the Obstacle.create
function. The first str in the tuple is always the main name, and the second
the short form of that name. The remainder are variants that are recognized but
their usage is not encouraged.

Attributes:
    circle_names (tuple[str]): Names associated with the circle shape
    rect_names (tuple[str]): Names associated with the rectangular shape
    half_circle_names (tuple[str]): Names associated with the half circle shape
    triangle_names (tuple[str]): Names associated with the triangle shape
    moon_names (tuple[str]): Names associated with the moon shape
    bullet_names (tuple[str]): Names associated with the bullet shape
    none_names (tuple[str]): Names associated with no shape
    names_list (tuple[tuple[str]]): Tuple of all name tuples.
"""
import numpy as np

from skimage.draw import circle, polygon
from scipy.ndimage.filters import minimum_filter, maximum_filter

import pcs_aero.util as util

# Just for convenience
_and, _not = np.logical_and, np.logical_not

# Attributes
circle_names = ('circle', 'c', 'sphere')
rect_names = ('rectangle', 'rect', 'r', 'square', 'block', 'b', 'cube')

half_circle_names = ('half-circle', 'halfcircle', 'hc', 'semi-circle',
                     'semicircle', 'sc', 'half-sphere', 'halfsphere', 'hs',
                     'semi-sphere', 'semisphere', 'ss')
triangle_names = ('triangle', 't', 'cone')
moon_names = ('moon', 'm')
bullet_names = ('bullet', 'b')
none_names = ('none', 'n', 'empty', '0')
names_lists = (circle_names, rect_names, half_circle_names, triangle_names,
               moon_names, bullet_names, none_names)


# Internal functions.
def _set_circle(mask, cx, cy, d, val=True):
    # Circle can simply be drawn with skimage
    ox, oy = circle(cx, cy, d)
    mask[ox, oy] = val


def _set_rect(mask, cx, cy, d, w, theta, val=True, left=None):
    l = r = w
    if left != None:
        l = left
    # Define rectangle coords and rotate them
    x = np.array([[cx - l, cx - l, cx + r, cx + r],
                  [cy - d, cy + d, cy + d, cy - d]])
    x = util.rotate(x, theta, (cx, cy))

    # Draw polygon with skimage
    ox, oy = polygon(x[0], x[1])
    mask[ox, oy] = val


def _set_triangle(mask, cx, cy, d, w, theta, val=True):
    # Horiziontal offset
    xo = w / 2

    # Define coords and rotate them
    x = np.array([[cx - xo, cx + xo, cx + xo], [cy, cy + d, cy - d]])
    x = util.rotate(x, theta, (cx, cy))

    # Draw polygon with skimage
    ox, oy = polygon(x[0], x[1])
    mask[ox, oy] = val


class Obstacle(object):
    """Obstacles define the mask of cells that are solid in the lbm
    simulation.

    Note: Obstacles should be created with the Obstacle.create function, it is
        possible to call the constructor directly, but in that case you have to
        create your own mask.
    """

    def __init__(self, mask):
        """Initialize the obstacle.

        Note: Calling this directly is not recommended, call Obstacle.create
            instead.

        Arguments:
            bool[w, h]: Mask of the obstacle.
        """
        self._nx, self._ny = mask.shape

        self._mask = mask

        self._ib = _and(self.mask, _not(minimum_filter(self.mask, 3)))
        self._ob = _and(maximum_filter(self.mask, 3), _not(self.mask))

        w = np.where(self.mask)[1]
        self._D = np.max(w) - np.min(w) + 1

    @property
    def shape(self):
        """(int, int): Dimensions of the mask."""
        return (self._nx, self._ny)

    @property
    def mask(self):
        """bool[*self.shape]: The actual boolean array."""
        return self._mask

    @property
    def inner_border(self):
        """bool[*self.shape]: Mask that only allows the inner border of the
        obstacle."""
        return self._ib

    @property
    def outer_border(self):
        """bool[*self.shape]: Mask that only allows the outer border of the
        obstacle."""
        return self._ob

    @property
    def D(self):
        """int: Height of the obstacle in cells."""
        return self._D

    @classmethod
    def create(cls, name, nx, ny, X=2 / 5, Y=0.5, D=0.125, W=0.125, theta=0):
        """Create a new obstacle by its name, the domain dimensions and it its
        properties.

        Arguments:
            name: Name of the obstacle, e.g. 'circle' or 'triangle'
            nx: number of cells in a row
            ny: number of cells in a column
            X: x-center of the obstacle as proportion of nx
            Y: y-center of the obstacle as proportion of ny
            D: diameter / height of the object as propertion of ny
            W: width of the object as propertion of nx
            theta: angle of aproach in degrees

        Returns:
            Obstacle

        Note:
            D, W are not necessarily the height of width of the object, e.g.
            a triangle is only half the width of W.

        """
        mask = np.zeros((nx, ny), dtype=bool)

        theta = theta * (np.pi / 180)

        # Calculate coords to grid coords
        cx = (nx - 1) * X
        cy = (ny - 1) * Y
        d = D * ny / 2
        w = W * ny / 2

        if name in circle_names:
            _set_circle(mask, cx, cy, d)

        elif name in rect_names:
            _set_rect(mask, cx, cy, d, w, theta)

        elif name in triangle_names:
            _set_triangle(mask, cx, cy, d, w, theta)

        elif name in half_circle_names:
            # A circle is created by creating a circle and removing a
            # rectangular polygon which is rotated.
            _set_circle(mask, cx, cy, d)
            _set_rect(mask, cx, cy, d, w, theta, val=False, left=0)

        elif name in moon_names:
            # The moon like shape is created with a circle, removing a inner
            # smaller circle and removing half of the remainder with a
            # rectangular polyong as with the half circle.
            _set_circle(mask, cx, cy, d)
            _set_circle(mask, cx, cy, d * 0.5, False)
            _set_rect(mask, cx, cy, d, w, theta, val=False, left=0)

        elif name in bullet_names:
            # Bullet is simple a circle plus a half rectangle
            _set_circle(mask, cx, cy, d)
            _set_rect(mask, cx, cy, d, w, theta, left=0)

        elif name in none_names:
            pass

        else:
            raise ValueError("Unknown obstacle name", name)

        return cls(mask)
