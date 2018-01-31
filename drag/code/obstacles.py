"""Defines the obstacle creation"""
import numpy as np

from skimage.draw import circle, polygon
from scipy.ndimage.filters import minimum_filter, maximum_filter

import code.util as util

# Just for convenience
l_and, l_not = np.logical_and, np.logical_not

# Collection of names that can be passed to the Obstacle.create function
# i.e. the names that can be passed by command line arguments
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


def set_circle(mask, cx, cy, d, val=True):
    # Circle can simply be drawn with skimage
    ox, oy = circle(cx, cy, d)
    mask[ox, oy] = val


def set_rect(mask, cx, cy, d, w, theta, val=True, left=None):
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


def set_triangle(mask, cx, cy, d, w, theta, val=True):
    # Horiziontal offset
    xo = w / 2

    # Define coords and rotate them
    x = np.array([[cx - xo, cx + xo, cx + xo], [cy, cy + d, cy - d]])
    x = util.rotate(x, theta, (cx, cy))

    # Draw polygon with skimage
    ox, oy = polygon(x[0], x[1])
    mask[ox, oy] = val


class Obstacle(object):
    "Obstacles define the mask of cells that are solid in the lbm simulation."

    def __init__(self, mask):
        self.nx, self.ny = mask.shape

        self.mask = mask

        self.inner_border = l_and(self.mask,
                                  l_not(minimum_filter(self.mask, 3)))
        self.outer_border = l_and(
            maximum_filter(self.mask, 3), l_not(self.mask))

        w = np.where(self.mask)[1]
        self.D = np.max(w) - np.min(w) + 1

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
            set_circle(mask, cx, cy, d)

        elif name in rect_names:
            set_rect(mask, cx, cy, d, w, theta)

        elif name in triangle_names:
            set_triangle(mask, cx, cy, d, w, theta)

        elif name in half_circle_names:
            # A circle is created by creating a circle and removing a
            # rectangular polygon which is rotated.
            set_circle(mask, cx, cy, d)
            set_rect(mask, cx, cy, d, w, theta, val=False, left=0)

        elif name in moon_names:
            # The moon like shape is created with a circle, removing a inner
            # smaller circle and removing half of the remainder with a
            # rectangular polyong as with the half circle.
            set_circle(mask, cx, cy, d)
            set_circle(mask, cx, cy, d * 0.5, False)
            set_rect(mask, cx, cy, d, w, theta, val=False, left=0)

        elif name in bullet_names:
            # Bullet is simple a circle plus a half rectangle
            set_circle(mask, cx, cy, d)
            set_rect(mask, cx, cy, d, w, theta, left=0)

        elif name in none_names:
            pass

        else:
            raise ValueError("Unknown obstacle name", name)

        return cls(mask)