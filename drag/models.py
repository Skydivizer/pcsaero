import abc

import numpy as np
import numexpr as ne
import sympy as sp

import obstacles
from constants import *

# Convenience for numpy multiplication
_w = w[:, np.newaxis, np.newaxis]


class Model(abc.ABC):
    """Base class for all models"""
    def __init__(self, Re=100, resolution=128, obstacle='circle', theta=0):
        self.Re = Re
        self.N = resolution
        self.shape = (self.N, self.N)
        self._obstacle = obstacles.Obstacle.create(obstacle, *self.shape, theta=theta)

        self.init()

    @property
    def obstacle(self):
        "Returns a boolean mask of which grid cells contain the obstacle"
        return self._obstacle.mask

    @property
    @abc.abstractmethod
    def population(self):
        "Returns a (9, *shape)-array of the population distribution function"
        pass

    @property
    @abc.abstractmethod
    def density(self):
        "Returns a (*shape)-array of the density"
        pass

    @property
    @abc.abstractmethod
    def velocity(self):
        "Returns a (*shape)-array of the absolute velocity"
        pass

    @property
    @abc.abstractmethod
    def time(self):
        "Returns the current time of the simulation"
        pass

    @property
    @abc.abstractmethod
    def equilibrium(self):
        "Returns a (9, *shape)-array of the population equilibrium"
        pass

    @property
    @abc.abstractmethod
    def drag_coefficient(self):
        "Returns the drag coefficient of the obstacle."
        pass

    @abc.abstractmethod
    def step(self):
        "Perform a single simulation step."
        pass

    @abc.abstractmethod
    def reset(self):
        "Reset the model to the initial state."
        pass

    @abc.abstractmethod
    def init(self):
        "Should be overriden in predecessor classes to initialize the object"
        pass


class SRT(Model):

    def init(self):
        self.Uin = 0.1
        self.nu = self.Uin * self.N / (self.Re * 8)

        self.dx = 1 / self.N
        self.dt = self.dx
        # self.tau = (self.dt * 3 / (self.dx**2 / self.nu)) + 0.5
        self.omega = 1 / (3 * self.nu + 0.5)
        # self.tau = 1 / self.omega

        self.reset()

    def reset(self):
        self.t = 0
        self.u = np.zeros((2, *self.shape), dtype=np.float64)
        self.u[0] = self.Uin

        self.rho = np.ones(self.shape, dtype=np.float64)

        self.update_equilibrium()
        self.f = self.feq

    def step(self):
        self.t += self.dt
        fin = self.f

        fin[col_W, -1, 1:-1] = fin[col_W, -2, 1:-1]
        self.update_macroscopic()
        self.u[0, 0, 1:-1] = self.Uin
        self.u[1, 0, 1:-1] = 0
        self.rho[0,1:-1] = 1 / (1 - self.Uin) * (np.sum(fin[col_C, 0, 1:-1], axis=0) + 2 *np.sum(fin[col_W,0,1:-1], axis=0))
        self.update_equilibrium()
        fin[col_E, 0, 1:-1] = self.feq[col_E, 0, 1:-1] + fin[col_Em,0,1:-1] - self.feq[col_Em,0,1:-1]
        self.collide()
        fout = self.f

        for k in range(9):
            fout[k, self.obstacle] = fin[idx_M[k], self.obstacle]
            fout[k, :, 0] = fin[idx_M[k], :, 0]
            fout[k, :, -1] = fin[idx_M[k], :, -1]

        self.stream()

    def collide(self):

        f = self.f
        omega = self.omega
        feq = self.feq

        self.f = f + omega * (feq - f)

    def stream(self):
        f = self.f

        for i in range(9):
            f[i,:,:] = np.roll(np.roll(f[i,:,:], e[i,0], axis=0), e[i,1], axis=1)

    def update_equilibrium(self):
        rho = self.rho
        u = self.u

        rho = self.rho
        u0, u1 = self.u[0], self.u[1]
        eu = np.tensordot(e, self.u, axes=([1], [0]))

        self.feq = rho * _w * (1.0 + (3) * eu + (9) * (eu**2) - (3) *
                               (u0**2 + u1**2))

    def update_macroscopic(self):
        self.rho = np.sum(self.f, axis=0)
        self.u = np.tensordot(e, self.f, axes=[[0], [0]]) / self.rho

    @property
    def population(self):
        return self.f

    @property
    def density(self):
        return self.rho

    @property
    def velocity(self):
        return np.sqrt(np.sum(self.u**2, axis=0))

    @property
    def time(self):
        return self.t

    @property
    def equilibrium(self):
        return self.feq

    @property
    def drag_coefficient(self):
        Fx = self.force[0]
        Uin = self.Uin
        D = self._obstacle.D
        return np.abs(Fx) / (Uin**2 * D)

    @property
    def lift_coefficient(self):
        Fy = self.force[1]
        Uin = self.Uin
        D = self._obstacle.D
        return Fy / (Uin**2 * D)

    @property
    def force(self):
        f = self.f
        s = np.zeros(2)
        
        for cell in np.argwhere(self._obstacle.inner_border):
            cx, cy = cell
            for k in range(1, 9):
                cxx, cyy = cell - e[k]
                if not self._obstacle.mask[cxx, cyy]:
                    s += e[k] * (f[idx_M[k], cxx, cyy] + f[k, cx, cy])

        return s


class MRT(Model):
    def init(self):

        self.Uin = 0.1
        self.dt = self.dx = 1 / self.N
        self.nu = self.Uin * self.N / self.Re
        self.omega = 1 / (3 * self.nu + 0.5)

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

        s_even = self.omega
        s_odd = 2 - self.omega
        self.S = np.array([0, s_even, s_even, 0, s_odd, 0, s_odd, s_even, s_even])[:,np.newaxis, np.newaxis]
        self.uin = np.fromfunction(lambda y: self.Uin * (4 / self.N**2) * y * (self.N - y), (self.N,))[1:-1]

        self.reset()


    def reset(self):
        self.t = 0
        self.m = np.zeros((9, *self.shape))
        self.m[0] = 1
        self.m[3, :, 1:-1] = self.uin

        self.m[3, self.obstacle] = 0

        self.update_equilibrium()
        self.f = np.tensordot(self.invM, self.meq, axes=[1, 0])

        

    def step(self):
        self.t += self.dt

        fin = self.f

        # Outlet condition neumann variant
        fin[col_W, -1, 1:-1] = (2 * fin[col_W, -2, 1:-1] + fin[col_W, -3, 1:-1]) / 3

        # Calculate moments
        self.m = np.tensordot(self.M, self.f, axes=[1, 0])

        # Inlet condition const velocity
        self.m[0, 0, 1:-1] = 1 / (1 - self.uin) * (np.sum(fin[col_C, 0, 1:-1], axis=0) + 2 *np.sum(fin[col_W,0,1:-1], axis=0))
        self.m[3, 0, 1:-1] = (self.uin / self.m[0, 0, 1:-1]) * (1 + 1e-3 * np.random.random(self.N - 2))
        self.m[5, 0, 1:-1] = 0

        # Calculate moment equilibrium
        self.update_equilibrium()

        # Collide by moments
        self.collide()

        # Calculate population
        self.f = np.tensordot(self.invM, self.m, axes=[1, 0])

        # Bounce back on obstacle
        fout = self.f
        for k in range(1,9):
            fout[k, self._obstacle.inner_border] = fin[idx_M[k], self._obstacle.inner_border]

        # Bounce back on north and south wall
        for k in row_N:
            fout[k, :, -1] = fin[idx_M[k], :, -1]
            fout[idx_M[k], :, 0] = fin[k, :, 0]
        
        # Transport populations
        self.stream()

    def update_equilibrium(self):
        eq = self.meq = np.empty(self.m.shape, dtype=np.float64)

        rho = self.m[0]
        jx = self.m[3]
        jy = self.m[5]

        eq[0] = rho
        eq[1] = -2 * rho + 3 * (jx**2 + jy**2)
        eq[2] = rho - 3 * (jx**2 + jy**2)
        eq[3] = jx
        eq[4] = -jx
        eq[5] = jy
        eq[6] = -jy
        eq[7] = jx**2 - jy**2
        eq[8] = jx * jy

    def collide(self):
        m = self.m
        S = self.S
        meq = self.meq
        self.m = ne.evaluate('m - S * (m - meq)')

    def stream(self):
        f = self.f

        for i in range(9):
            f[i,:,:] = np.roll(np.roll(f[i,:,:], e[i,0], axis=0), e[i,1], axis=1)

    @property
    def density(self):
        return self.m[0]

    @property
    def velocity(self):
        u = self.m[[3,5]] / self.m[0]
        return np.sqrt(np.sum(u**2, axis=0))

    @property
    def time(self):
        return self.t

    @property
    def equilibrium(self):
        return self.meq
    @property
    def population(self):
        return self.f

    @property
    def drag_coefficient(self):
        Fx = self.force[0]
        Uin = self.Uin 
        D = self._obstacle.D
        return np.abs(Fx) / (0.5 * Uin**2 * D)

    @property
    def lift_coefficient(self):
        Fy = self.force[1]
        Uin = self.Uin 
        D = 1/8 * self.N
        return Fy / (0.5 * Uin**2 * D)

    @property
    def force(self):
        "Returns the force exchange between the fluid and the solid obstacle."
        # f = self.f
        rho = self.m[0]
        s = np.zeros(2)
        
        # For every border cell in the bostacle
        for cell in np.argwhere(self._obstacle.inner_border):
            cx, cy = cell

            # For each fluid cell that streams into this solid cill
            for k in range(1, 9):
                cxx, cyy = cell - e[k]
                if not self._obstacle.mask[cxx, cyy]:
                    # Add population to the force sum                  # Fld|Sld
                    # s += e[k] * (f[k, cx, cy] + f[idx_M[k], cxx, cyy]) # <- | ->  # Described by literature
                    s += w[k] * e[k] * (rho[cxx, cyy]) # -> | <-  # Seems more intuitive?

        return s * self.nu


class PyLBM(Model):
    # This is just a model for reference. It does not fully implement the interface
    # Also it seems to leak populations between solids and fluids, which seems strange..
    def init(self):
        self.obstacle = np.zeros((*self.shape), dtype=bool)
        try:
            import pyLBM

        except ImportError:
            print("You need the pyLBM package to run this model.")
            quit()

        rayon = 1/16
        Re = self.Re
        dx = 1 / self.N
        la = 1
        v0 = 0.1
        rhoo = 1
        eta = nu = rhoo * v0 * 2 * rayon / Re
        mu = 1.5e-3

        X, Y, LA = sp.symbols('X, Y, LA')
        rho, qx, qy = sp.symbols('rho, qx, qy')

        def bc_in(f, m, x, y):
            m[qx] = rhoo * v0

        xmin, xmax, ymin, ymax = 0., 1., 0., 1.

        dummy = 3.0/(la*rhoo*dx)
        s_mu = 1.0/(0.5+mu*dummy)
        s_eta = 1.0/(0.5+eta*dummy)
        s_q = s_eta
        s_es = s_mu
        s  = [0.,0.,0.,s_mu,s_es,s_q,s_q,s_eta,s_eta]

        dummy = 1./(LA**2*rhoo)
        qx2 = dummy*qx**2
        qy2 = dummy*qy**2
        q2  = qx2+qy2
        qxy = dummy*qx*qy


        dico = {
            'box':{'x':[xmin, xmax], 'y':[ymin, ymax], 'label':[0,2,0,0]},
            'elements':[pyLBM.Circle([1/3, 1/2], rayon, label=1)],
            'space_step':dx,
            'scheme_velocity':la,
            'parameters':{LA:la},
            'schemes':[
                {
                    'velocities':list(range(9)),
                    'conserved_moments':[rho, qx, qy],
                    'polynomials':[
                        1, LA*X, LA*Y,
                        3*(X**2+Y**2)-4,
                        (9*(X**2+Y**2)**2-21*(X**2+Y**2)+8)/2,
                        3*X*(X**2+Y**2)-5*X, 3*Y*(X**2+Y**2)-5*Y,
                        X**2-Y**2, X*Y
                    ],
                    'relaxation_parameters':s,
                    'equilibrium':[
                        rho, qx, qy,
                        -2*rho + 3*q2,
                        rho-3*q2,
                        -qx/LA, -qy/LA,
                        qx2-qy2, qxy
                    ],
                    'init':{rho:rhoo, qx:0., qy:0.},
                },
            ],
            'boundary_conditions':{
                0:{'method':{0: pyLBM.bc.Bouzidi_bounce_back}, 'value':bc_in},
                1:{'method':{0: pyLBM.bc.Bouzidi_bounce_back}, 'value':None},
                2:{'method':{0: pyLBM.bc.Neumann_x}, 'value':None},
            },
            'generator': pyLBM.generator.NumpyGenerator,
        }

        self.sim = pyLBM.Simulation(dico)


    @property
    def population(self):
        return self.sim.F[:]

    @property
    def density(self):
        return self.sim.m[0, :]

    @property
    def velocity(self):
        rho = self.density
        u = self.sim.m[[1,2], :] / rho
        return np.sqrt(np.sum(u**2, axis=0))

    @property
    def time(self):
        return self.sim.t

    @property
    def equilibrium(self):
        return self.sim.m[:]
    
    @property
    def drag_coefficient(self):
        return "N/A"

    def step(self):
        self.sim.one_time_step()

    def reset(self):
        pass

