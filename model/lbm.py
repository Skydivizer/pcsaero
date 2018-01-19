"""A simple Lattice boltzmann method implementation"""
# Basically the same as
# http://cui.unige.ch/~chopard/FTP/USI/d2q9.py
import numpy as np
import numexpr as ne

v = np.array([ [ 1,  1], [ 1,  0], [ 1, -1], [ 0,  1], [ 0,  0],
            [ 0, -1], [-1,  1], [-1,  0], [-1, -1] ])
t = np.array([ 1/36, 1/9, 1/36, 1/9, 4/9, 1/9, 1/36, 1/9, 1/36])[:,np.newaxis, np.newaxis]

col3 = np.array([6, 7, 8])

class Model():
    @property
    def velocity(self):
        return np.sqrt(np.sum(self.u**2, axis=0))

    @property
    def density(self):
        return self.rho

    @property
    def population(self):
        return self.fin

    @property
    def equilibrium(self):
        return self.feq

    @property
    def drag(self):
        density = self.rho_in
        diameter = self.shape[1] / 6
        return np.abs(self.force_on_obstacle[0]) / (0.5 * density * np.mean(self.u[0, 0])**2 * diameter)

    @property
    def force_on_obstacle(self):
        s = 0
        for cell in np.argwhere(self.boundary):
            for k in range(9):
                e = v[8 - k] 
                ce = cell + e
                if not self.obstacle[ce[0], ce[1]]:
                    s += e * (self.fin[k, cell[0], cell[1]] + self.fin[8-k, ce[0], ce[1]])

        return s

    def __init__(self, Re=220, shape=(210, 90), obstacle='circle'):

        self.Re = Re
        self.shape = shape
        self.uLB = 0.04

        r = shape[1] / 6

        self.nuLB = self.uLB * r / self.Re
        self.omega = 1 / (3 * self.nuLB + 0.5)

        self.rho_in = 1.05
        self.rho_out = 1

        cx = self.shape[0] / 3
        cy = self.shape[1] / 2
        if obstacle == 'circle':
            def fun(x, y):
                return (x - cx)**2 + (y - cy)**2 < r**2
        elif obstacle == 'square':
            def fun(x, y):
                return np.logical_and(np.abs(x - cx) < r, np.abs(y- cy) < r)
        elif obstacle == 'wall':
            def fun(x, y):
                return np.abs(x - cx) < r
        elif obstacle == 'none':
            def fun(x, y):
                return x != x

        self.obstacle = np.fromfunction(fun, self.shape)

        oy = np.diff(self.obstacle.astype(int), axis=0)
        woy = np.where(oy == -1)

        ox = np.diff(self.obstacle.astype(int), axis=1)
        wox = np.where(ox == -1)

        boundary = np.zeros(self.obstacle.shape, dtype=bool)
        boundary[wox[0], wox[1]-1] = True
        boundary[woy[0] - 1, woy[1]] = True
        boundary[np.where(oy == 1)] = True
        boundary[np.where(ox == 1)] = True

        self.boundary = boundary

        def inivel(d, x, y):
            return (1-d) * self.uLB * (1 + 1e-4*np.sin(y/(self.shape[1]-1)*2*np.pi))
        self.vel = np.fromfunction(inivel, (2, *self.shape))
        self.vel[:, 1:, :] *= 0.5

        self.reset()

    def update_macroscopic(self):
        self.rho = np.sum(self.fin, axis=0)
        self.u = np.tensordot(v, self.fin, axes=[[0], [0]]) / self.rho

        self.u[:, self.obstacle] = 0
        self.rho[self.obstacle] = 0

    def update_equilibrium(self):
        rho = self.rho
        u0, u1 = self.u[0], self.u[1]

        cu = 3.0 * np.tensordot(v, self.u, axes=([1], [0]))
        
        self.feq = (ne.evaluate("rho * t * (1.0 + cu + 0.5 * (cu ** 2) - (3.0 / 2.0 * (u0**2 + u1**2)))"))

    def step(self):
        # Right wall: outflow condition.
        self.fin[col3,-1,:] = self.fin[col3,-2,:]

        # Compute macroscopic variables, density and velocity.
        self.update_macroscopic()

        # Left wall: inflow condition.
        self.u[:,0,:] = self.vel[:,0,:]
        # self.rho[0,:] = self.rho_in

        # Right wall again
        # self.u[:,-1,:] = self.vel[:,-1,:]
        self.rho[-1, :] = self.rho_out

        # Compute equilibrium.
        self.update_equilibrium()
        # self.fin[[0,1,2],0,:] = self.feq[[0,1,2],0,:] + self.fin[[8,7,6],0,:] - self.feq[[8,7,6],0,:]

        # Does not behave nice with setting rho and u manually
        # self.fin[[6,7,8],-1,:] = self.feq[[6,7,8],-1,:] + self.fin[[2,1,0],-1,:] - self.feq[[6,7,8],-1,:]

        # Collision step.
        fin = self.fin
        omega = self.omega
        feq = self.feq

        fout = ne.evaluate("fin + omega * (feq - fin)")

        # Bounce-back condition for obstacle.
        for i in range(9):
            fout[i, self.obstacle] = fin[8-i, self.obstacle]

        # Streaming step.
        for i in range(9):
            self.fin[i,:,:] = np.roll(np.roll(fout[i,:,:], v[i,0], axis=0), v[i,1], axis=1)

    def reset(self):
        self.rho = np.repeat(np.linspace(self.rho_in, self.rho_out, self.shape[0])[:, np.newaxis], self.shape[1], axis=1)
        self.u = self.vel
        self.update_equilibrium()
        self.fin = self.feq
