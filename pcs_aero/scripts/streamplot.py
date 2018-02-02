#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

import pcs_aero.args as args
import pcs_aero.models as models


def make_UV(model):
    U = (model.m[3] / model.m[0]).T[::-1]
    V = (model.m[5] / model.m[0]).T
    Y, X = np.mgrid[0:model.N, 0:model.N]

    return X, Y, U, V


if __name__ == "__main__":
    parser = args.ModelArgParser(description="Create a streamplot of some "
                                 "simulation.")

    parser.add_argument('file_name', type=str, help='File name to save as.')

    args, model = parser.parse_args()

    name = args.file_name

    X, Y, U, V = make_UV(model)
    o = model.obstacle_mask[:, ::-1].T
    omask = np.ma.masked_where(~o, np.ones(o.shape))

    U = np.ma.array(U, mask=o)
    V = np.ma.array(V, mask=o)

    speed = model.velocity.T[::-1]

    plt.xlabel('x')
    plt.ylabel('y')
    plt.xticks([])
    plt.yticks([])
    plt.title('Stream velocities for {shape} with $\\theta={theta:3}$'.format(
        shape=model.obstacle['name'],
        theta=model.obstacle['theta']))
    strm = plt.streamplot(
        X, Y, U, V, density=2, linewidth=1, color=speed, cmap='YlOrRd')
    cbar = plt.colorbar(strm.lines)

    plt.imshow(omask, cmap='binary', alpha=1, vmin=0, vmax=1)

    plt.savefig(name)
