import numpy as np
from scipy.ndimage.filters import minimum_filter, maximum_filter

l_and, l_not = np.logical_and, np.logical_not

circle_names = ('circle', 'c', 'sphere')
rect_names = ('rectangle', 'rect', 'r', 'square', 'block', 'b', 'cube')
half_circle_names = ('half-circle', 'halfcircle', 'hc', 'semi-circle',
                     'semicircle', 'sc', 'half-sphere', 'halfsphere', 'hs',
                     'semi-sphere', 'semisphere', 'ss')
angled_square_names = ('angled-square', 'as', 'a', 'angledsquare',
                       'angled-block', 'angledblock', 'ab', 'angled-cube',
                       'angledcube', 'ac')
triangle_names = ('triangle', 't', 'cone')
none_names = ('none', 'n', 'empty', '0')

names_lists = (circle_names, rect_names, half_circle_names,
               angled_square_names, triangle_names, none_names)
names = [l[0] for l in names_lists]
names_map = {n: l[0] for l in names_lists for n in l}


def names_type(string):
    string = str.lower(string)
    try:
        return names_map[string]
    except KeyError:
        raise TypeError(string, 'not a known obstacle name')


class Obstacle(object):
    def __init__(self, nx, ny, func=None):

        self.nx = nx
        self.ny = ny

        if func == None:
            self.mask = np.zeros((self.nx, self.ny), dtype=bool)
        else:
            self.mask = np.fromfunction(func, (self.nx, self.ny))

        self.inner_border = l_and(self.mask,
                                  l_not(minimum_filter(self.mask, 3)))
        self.outer_border = l_and(
            maximum_filter(self.mask, 3), l_not(self.mask))

    def add(self, other):
        self.mask |= other.mask

    @classmethod
    def create(cls, name, nx, ny, X=1 / 3, Y=1 / 2, D=1 / 8, W=1 / 8):
        cx = nx * X
        cy = ny * Y
        d = D * ny / 2
        w = W * ny / 2

        name = name

        if name in rect_names:

            def func(x, y):
                return l_and(np.abs(x - cx) < w, np.abs(y - cy) < d)
        elif name in circle_names:

            def func(x, y):
                return (x - cx)**2 + (y - cy)**2 <= d**2
        elif name in half_circle_names:

            def func(x, y):
                return l_and((x - cx)**2 + (y - cy)**2 <= d**2, x - cx <= 0)
        elif name in triangle_names:

            def func(x, y):
                lx = cx - w
                return l_and(np.abs(x - cx) < w, np.abs(y - cy) < (x - lx) / 2)

        elif name in angled_square_names:

            def func(x, y):
                lx = cx - w
                rx = cx + w
                dy = np.abs(y - cy)
                return l_and(
                    np.abs(x - cx) < w, l_and(dy < (x - lx), dy < (rx - x)))

        elif name in none_names:
            func = None
        else:
            raise ValueError("Unknown obstacle name", name)

        return cls(nx, ny, func)