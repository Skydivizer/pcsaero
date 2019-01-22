# -*- coding: utf-8 -*-
"""This module defines the lbm models that can be used to run the simulation.

All models are subclasses of the abstract Model class, implenting this
interface should guarantee that a model works with all the scripts used in this
project.

The three implemented models are a SRT, a TRT, and a MRT model. Currently the
MRT and TRT model are exactly the same. Work has to be done on choosing the
relaxation paramters in the MRT model.

Note:
    The characteristic length is set to unity for all models.
"""

import abc
import functools

import numpy as np
import numexpr as ne

import pcs_aero.obstacles as obstacles
from pcs_aero.constants import *

# Convenience
_w = w[:, np.newaxis, np.newaxis]
_sum0 = lambda x: np.sum(x, axis=0)


class Model(abc.ABC):
    """Abstract base class for all models.

    To implemented this class three properties (population, time, equilibrium),
    and two functions (step, reset) need to be implemented.
    """

    def __init__(self,
                 Re=100,
                 resolution=128,
                 obstacle='circle',
                 theta=0,
                 size=0.125,
                 Uin=0.1):
        """Instantiate a new model.

        Arguments:
            Re (float): The reynolds number to use in the model.
            resolution (int): Number of cells per characteristic length.
            obstacle (str): Name of the obstacle to use, e.g. 'circle'.
            theta (float): Counter clockwise rotation of the obstacle in
                degrees.
            size (float): Diameter of the obstacle in characteristic length.
            Uin (float): Max stream velocity at inlet in the lattice units.
        """
        self._Re = Re
        self._N = resolution
        self._Uin = Uin
        self._uin = np.fromfunction(
            lambda y: self.Uin * (4 / self.N**2) * y * (self.N - y),
            (self.N, ))[1:-1]
        self._shape = (self.N, self.N)
        self._obstacle = obstacles.Obstacle.create(
            obstacle, *self.shape, D=size, W=size, theta=theta)
        self._dt = self._dx = 1 / self.N
        self._nu = self.Uin * self.N / self.Re

        self._obstacle_info = {
            'name': obstacle,
            'theta': theta,
            'size': size,
        }

    @property
    def Re(self):
        """float: Reynolds number used in the model."""
        return self._Re

    @property
    def N(self):
        """int: Resolution used in the model, i.e. the number of cells per
        characteristic length."""
        return self._N

    @property
    def Uin(self):
        """float: Max stream velocity at inlet in lattice units."""
        return self._Uin

    @property
    def shape(self):
        """(int, int): Number of cells in vertical and horizontal direction."""
        return self._shape

    @property
    def obstacle(self):
        """dict(str, object): The properties of the obstacle used in the
        model.

        The returned dictionary contains three keys: name, theta, and size.

        Example:
        {
            'name': 'circle',
            'theta': 0,
            'size': 0.125,
        }
        """
        return self._obstacle_info

    @property
    def obstacle_mask(self):
        """bool[*self.shape]: Mask of which grid cells contain the obstacle."""
        return self._obstacle.mask

    @property
    def dt(self):
        """float: Delta time for one time step."""
        return self._dt

    @property
    def dx(self):
        """float: Delta distance for one time step."""
        return self._dx

    @property
    def nu(self):
        """float: Viscosity used in the the model in lattice units."""
        return self._nu

    @property
    def density(self):
        """float[*self.shape]: The matrix of cell densities."""
        return _sum0(self.population)

    @property
    def velocity(self):
        """float[*self.shape]: The matrix of cell velocities."""
        u = np.tensordot(e, self.f, axes=[[0], [0]]) / self.rho
        return u / self.density

    @property
    def force(self):
        """(float, float): The force vector action on the obstacle."""
        rho = self.density
        s = np.zeros(2)

        # For every border cell in the bostacle
        for cell in np.argwhere(self._obstacle.inner_border):
            cx, cy = cell

            # For each fluid cell that streams into this solid cill
            for k in range(1, 9):
                cxx, cyy = cell - e[k]
                if not self._obstacle.mask[cxx, cyy]:
                    s += w[k] * e[k] * (rho[cxx, cyy])

        return s * self.nu

    @property
    def drag_coefficient(self):
        """float: The drag coefficient of the obstacle in the current
        stream."""
        Fx = self.force[0]
        Uin = self.Uin
        D = self._obstacle.D
        return np.abs(Fx) / (0.5 * Uin**2 * D)

    @property
    def lift_coefficient(self):
        """float: The lift coefficient of the obstacle in the current
        stream."""
        Fy = self.force[1]
        Uin = self.Uin
        D = self._obstacle.D
        return Fy / (0.5 * Uin**2 * D)

    @property
    @abc.abstractmethod
    def population(self):
        """float[*self.shape]: Matrix of the population distribution
        function."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def time(self):
        """float: Current simulation time."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def equilibrium(self):
        """float[*self.shape]: Matrix of the population equlibrium function."""
        raise NotImplementedError

    @abc.abstractmethod
    def step(self):
        """Continue the simulation by self.dt time."""
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self):
        """Reset the model to the initial state.

        Note: After resetting the model should be in the exact same state
            (or an equivalent state if the model is random) as it was after
            initializing."""
        raise NotImplementedError


class SRT(Model):
    """Single relaxation time model.

    The model picks the relaxation time based on the lattice viscosity, e.g.
    omega = 1 / (3 * nu + 0.5).
    """

    def __init__(self, *args, **kwargs):
        super(SRT, self).__init__(*args, **kwargs)

        self.omega = 1 / (3 * self.nu + 0.5)

        self.reset()

    # Implementations of abstract properties
    @property
    def population(self):
        return self.f

    @property
    def time(self):
        return self.t

    @property
    def equilibrium(self):
        return self.feq

    # Overriding some properties for efficiency.
    @property
    def density(self):
        return self.rho

    @property
    def velocity(self):
        return np.sqrt(_sum0(self.u**2))

    # Implemetation of abstract functions
    def step(self):
        self.t += self.dt
        fin = self.f

        # Outlet condition neumann variant
        fin[col_W, -1, 1:-1] = (
            2 * fin[col_W, -2, 1:-1] + fin[col_W, -3, 1:-1]) / 3

        self._update_macroscopic()

        # Inlet condition const velocity
        self.rho[0, 1:-1] = 1 / (1 - self._uin) * (
            _sum0(fin[col_C, 0, 1:-1]) + 2 * _sum0(fin[col_W, 0, 1:-1]))
        self.u[0, 0, 1:-1] = self._uin / self.rho[0, 1:-1]
        self.u[1, 0, 1:-1] = 0

        self._update_equilibrium()

        # Collide population
        self._collide()

        # Bounce back on obstacle
        fout = self.f
        for k in range(1, 9):
            msk = self._obstacle.inner_border
            fout[k, msk] = fin[idx_M[k], msk]

        # Bounce back on north and south wall
        for k in row_N:
            fout[k, :, -1] = fin[idx_M[k], :, -1]
            fout[idx_M[k], :, 0] = fin[k, :, 0]

        # Transport population
        self._stream()

    def reset(self):
        self.t = 0
        self.u = np.zeros((2, *self.shape), dtype=np.float64)
        self.u[0, :, 1:-1] = self._uin

        self.rho = np.ones(self.shape, dtype=np.float64)

        self.update_equilibrium()
        self.f = self.feq

    # Internal functions
    def _collide(self):

        f = self.f
        omega = self.omega
        feq = self.feq

        self.f = ne.evaluate('f + omega * (feq - f)')

    def _stream(self):
        f = self.f

        for i in range(9):
            f[i, :, :] = np.roll(
                np.roll(f[i, :, :], e[i, 0], axis=0), e[i, 1], axis=1)

    def _update_equilibrium(self):
        rho = self.rho
        u = self.u

        rho = self.rho
        u0, u1 = self.u[0], self.u[1]
        eu = np.tensordot(e, self.u, axes=([1], [0]))

        self.feq = rho * _w * (1.0 + (3) * eu + (9) * (eu**2) - (3) *
                               (u0**2 + u1**2))

    def _update_macroscopic(self):
        self.rho = _sum0(self.f)
        self.u = np.tensordot(e, self.f, axes=[[0], [0]]) / self.rho


class TRT(Model):
    """Two relaxation time model.

    For this model the relaxation parameters are based on the viscosity, e.g:
        omega_plus = 1 / (3 * nu + 0.5)
        omega_min = 2 - omega_plus

    The latter following from Lambda = 0.25 for maximum LBE stability.

    Note: This model essentially implemnts a MRT model with restricted 
        relaxtion parameters.
    """

    def __init__(self, *args, **kwargs):
        super(TRT, self).__init__(*args, **kwargs)

        # Create transformation matrices
        self.M = np.array([
            [ 1,  1,  1,  1,  1,  1,  1,  1,  1],
            [-4, -1, -1, -1, -1,  2,  2,  2,  2],
            [ 4, -2, -2, -2, -2,  1,  1,  1,  1],
            [ 0,  1,  0, -1,  0,  1, -1, -1,  1],
            [ 0, -2,  0,  2,  0,  1, -1, -1,  1],
            [ 0,  0,  1,  0, -1,  1,  1, -1, -1],
            [ 0,  0, -2,  0,  2,  1,  1, -1, -1],
            [ 0,  1, -1,  1, -1,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0,  1, -1,  1, -1]])

        self.invM = np.array([
            [1/9, -1/9,    1/9,    0,     0,    0,     0,    0,    0],
            [1/9, -1/36, -1/18,  1/6,  -1/6,    0,    -0,  1/4,    0],
            [1/9, -1/36, -1/18,    0,     0,  1/6,  -1/6, -1/4,    0],
            [1/9, -1/36, -1/18, -1/6,   1/6,    0,    -0,  1/4,    0],
            [1/9, -1/36, -1/18,   -0,    -0, -1/6,   1/6, -1/4,    0],
            [1/9,  1/18,  1/36,  1/6,  1/12,  1/6,  1/12,    0,  1/4],
            [1/9,  1/18,  1/36, -1/6, -1/12,  1/6,  1/12,    0, -1/4],
            [1/9,  1/18,  1/36, -1/6, -1/12, -1/6, -1/12,    0,  1/4],
            [1/9,  1/18,  1/36,  1/6,  1/12, -1/6, -1/12,    0, -1/4]])

        self._init_S()

        self.reset()

    # Implementations of abstract properties
    @property
    def time(self):
        return self.t

    @property
    def equilibrium(self):
        return self.meq

    @property
    def population(self):
        return self.f

    # Overriden properties for efficiency
    @property
    def density(self):
        return self.m[0]

    @property
    def velocity(self):
        u = self.m[[3, 5]] / self.m[0]
        return np.sqrt(_sum0(u**2))

    # Implementations of abstract methods
    # Note: Try to keep the logic the same as in SRT for possible comparision
    def step(self):
        self.t += self.dt
        fin = self.f

        # Outlet condition neumann variant
        fin[col_W, -1, 1:-1] = (
            2 * fin[col_W, -2, 1:-1] + fin[col_W, -3, 1:-1]) / 3

        # Calculate moments
        self.m = np.tensordot(self.M, self.f, axes=[1, 0])

        # Inlet condition const velocity
        self.m[0, 0, 1:-1] = 1 / (1 - self._uin) * (
            _sum0(fin[col_C, 0, 1:-1]) + 2 * _sum0(fin[col_W, 0, 1:-1]))
        self.m[3, 0, 1:-1] = self._uin / self.m[0, 0, 1:-1]
        self.m[5, 0, 1:-1] = 0

        # Calculate moment equilibrium
        self._update_equilibrium()

        # Collide by moments
        self._collide()

        # Calculate population
        self.f = np.tensordot(self.invM, self.m, axes=[1, 0])

        # Bounce back on obstacle
        fout = self.f
        for k in range(1, 9):
            msk = self._obstacle.inner_border
            fout[k, msk] = fin[idx_M[k], msk]

        # Bounce back on north and south wall
        for k in row_N:
            fout[k, :, -1] = fin[idx_M[k], :, -1]
            fout[idx_M[k], :, 0] = fin[k, :, 0]

        # Transport populations
        self._stream()

    def reset(self):
        self.t = 0

        # Initialize moments
        self.m = np.zeros((9, *self.shape))
        self.m[0] = 1
        self.m[3, :, 1:-1] = self._uin
        self.m[3, self.obstacle_mask] = 0

        # Get population equilibrium from moments equillibrium
        self._update_equilibrium()
        self.f = np.tensordot(self.invM, self.meq, axes=[1, 0])

    def _init_S(self):
        """Initialize diagonal relaxtion matrix S."""
        sp = 1 / (3 * self.nu + 0.5)
        sm = 2 - sp
        self.S = np.array([0, sp, sp, 0, sm, 0, sm, sp, sp])

        # For easier numpy multiplication
        self.S = self.S[:, np.newaxis, np.newaxis]

    # Internal functions
    def _update_equilibrium(self):
        eq = self.meq = np.empty(self.m.shape, dtype=np.float64)

        rho = self.m[0]
        jx = self.m[3]
        jy = self.m[5]
        jxsqr = jx**2
        jysqr = jy**2
        jxysqr3 = 3 * (jxsqr + jysqr)

        eq[0] = rho
        eq[1] = -2 * rho + jxysqr3
        eq[2] = rho - jxysqr3
        eq[3] = jx
        eq[4] = -jx
        eq[5] = jy
        eq[6] = -jy
        eq[7] = jxsqr - jysqr
        eq[8] = jx * jy

    def _collide(self):
        m = self.m
        S = self.S
        meq = self.meq
        self.m = ne.evaluate('m - S * (m - meq)')

    def _stream(self):
        f = self.f

        for i in range(9):
            f[i, :, :] = np.roll(
                np.roll(f[i, :, :], e[i, 0], axis=0), e[i, 1], axis=1)


class MRT(TRT):
    """Multiple relaxtion time model.

    Note: No logic implemented, thus the same as TRT.
    """

    def _init_S(self):

        # Change this to pick different relaxation parameters.
        super(MRT, self)._init_S()
