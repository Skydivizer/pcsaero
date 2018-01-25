import abc

import numpy as np
import sympy as sp

import obstacles
from constants import *

_w = w[:, np.newaxis, np.newaxis]


class Model(abc.ABC):
    def __init__(self, Re=100, resolution=128, obstacle='circle'):
        self.Re = Re
        self.N = resolution
        self.shape = (self.N, self.N)
        self._obstacle = obstacles.Obstacle.create(obstacle, self.N, self.N)

        self.init()

    @property
    def obstacle(self):
        return self._obstacle.mask

    @property
    @abc.abstractmethod
    def population(self):
        pass

    @property
    @abc.abstractmethod
    def density(self):
        pass

    @property
    @abc.abstractmethod
    def velocity(self):
        pass

    @property
    @abc.abstractmethod
    def time(self):
        pass

    @property
    @abc.abstractmethod
    def equilibrium(self):
        pass

    @property
    @abc.abstractmethod
    def drag_coefficient(self):
        pass

    @abc.abstractmethod
    def step(self):
        pass

    @abc.abstractmethod
    def reset(self):
        pass

    @abc.abstractmethod
    def init(self):
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
        self.u = np.zeros((2, self.N, self.N), dtype=np.float128)
        self.u[0] = self.Uin
        self.u[:, self.obstacle] = 0

        self.rho = np.ones((self.N, self.N), dtype=np.float128)

        self.update_equilibrium()
        self.f = self.feq

        self.f_in = self.f[:, 0].copy()

    def step(self):
        self.t += self.dt

        self.stream()
        self.update_macroscopic()

        # All the clipping is done to attempt stabilizing the sim..
        # Yet trouble still occurs somehow
        # self.rho = np.clip(self.rho, 0, 2)
        # self.u = np.clip(self.u, -1 / np.sqrt(3), 1 / np.sqrt(3))

        # Const inlet velocity + pressure
        self.u[0, 0, 1:-1] = self.Uin
        self.rho[0, :] = 1

        self.update_equilibrium()

        # Again
        self.feq = np.clip(self.feq, 0, 2)
        self.collide()

    def collide(self):

        f = self.f
        omega = self.omega
        feq = self.feq

        self.f = f + omega * (feq - f)

    def stream(self):

        fin = self.f.copy()
        f = self.f

        for i in range(9):
            f[i,:,:] = np.roll(np.roll(f[i,:,:], e[i,0], axis=0), e[i,1], axis=1)

        for k in range(9):
            f[k, self.obstacle] = fin[idx_M[k], self.obstacle]

            # f[k, :, 0] = fin[idx_M[k], :, 0]
            # f[k, :, -1] = fin[idx_M[k], :, -1]

        f[:, 0,  1:-1] = self.f_in[:, 1:-1] * ( 1 + 1e-3 * np.random.randn(self.N - 2) )
        f[:, -1, 1:-1] = fin[:, -2, 1:-1]

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
        D = 1/16 * self.N
        return np.abs(Fx) / (0.5 * Uin**2 * D)

    @property
    def force(self):
        f = self.f
        s = np.zeros(2)
        
        for cell in np.argwhere(self._obstacle.inner_border):
            cx, cy = cell
            for k in range(9):
                cxx, cyy = cell - e[k]
                if not self._obstacle.mask[cxx, cyy]:
                    s += e[k] * (f[k, cx, cy] + f[idx_M[k], cxx, cyy])

        return s * self.nu


class MRT(Model):
    def __init__(self):
        pass


class PyLBM(Model):
    def init(self):
        self.obstacle = np.zeros((self.N, self.N), dtype=bool)
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

