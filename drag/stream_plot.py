#!/usr/bin/env python3
"""Run this script to calculate the drag and life coefficient of 2d objects in
a 1x1 pipe.
"""
import numpy as np

import code.args as args
import code.models as models

import matplotlib.pyplot as plt


def make_UV(model):
    U = (model.m[3] / model.m[0]).T
    V = (model.m[5] / model.m[0]).T
    Y, X = np.mgrid[0:model.N, 0:model.N]

    return X, Y, U, V


if __name__ == "__main__":
    # Handle command line arguments
    parser = args.ModelArgParser()

    args, model = parser.parse_args()

    X, Y, U, V = make_UV(model)
    o = model.obstacle.T
    omask = np.ma.masked_where(~o, np.ones(o.shape))

    U = np.ma.array(U, mask=o)
    V = np.ma.array(V, mask=o)

    speed = model.velocity.T

    plt.xlabel('x')
    plt.ylabel('y')
    plt.xticks([])
    plt.yticks([])
    plt.title('Stream velocities')
    strm = plt.streamplot(X, Y, U, V, density=2, linewidth=1, color=speed, cmap='YlOrRd')
    cbar = plt.colorbar(strm.lines)
    # cbar.ax.set_yticklabels(['Low', 'High'])

    # plt.imshow(model.density.T, cmap='coolwarm')
    plt.imshow(omask, cmap='binary', alpha=1, vmin=0, vmax=1)

    plt.savefig('streamplot.pdf')
