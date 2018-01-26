#!/usr/bin/env python3
"""Run this script to calculate the drag coefficient of 2d objects in a 1x1
pipe.
"""

import argparse
import gui

import models
import obstacles

# To not overflow the help interface of arugment parser only the main obstacle
# names are # shown as possible options
onames = [l[0] for l in obstacles.names_lists]
onames_map = {n: l[0] for l in obstacles.names_lists for n in l}

# Defines the type for argument parser
def onames_type(string):
    string = str.lower(string)
    try:
        return onames_map[string]
    except KeyError:
        raise TypeError(string, 'not a known obstacle name')


if __name__ == "__main__":

    # Handle command line arguments
    parser = argparse.ArgumentParser(
        description="Program that tries to calculate the drag coefficient of "
        "2d objects in a 1x1 pipe.")
    parser.add_argument(
        "-m",
        "--model",
        help="The model to use. All models use lattice boltzmann methods, the "
        "difference lies in the implementation. [srt] single relaxation time. "
        "[mrt] multiple relaxation times [pylbm] mrt model that wraps pyLBM",
        type=str.lower,
        choices=['srt', 'mrt', 'pylbm'],
        default='srt')
    parser.add_argument(
        "-r",
        "--resolution",
        help='Number of cells per 1 characteristic length',
        type=int,
        default=64)
    parser.add_argument(
        "-R", "--reynolds", help='Reynolds number', type=float, default=500)
    parser.add_argument(
        "-o",
        "--obstacle",
        help='The obstacle to calculate the drag coefficient of.',
        type=onames_type,
        choices=onames,
        default=onames[0])

    parser.add_argument(
        "-t",
        "--theta",
        type=float,
        help="Angle of approach in degrees",
        default=0)

    parser.add_argument(
        '-T',
        '--time',
        type=float,
        help="Runs the simulation without a graphical user interface for a "
        "specified simulation time.")
    args = parser.parse_args()

    # Create the model
    model = {
        "srt": models.SRT,
        "mrt": models.MRT,
        "pylbm": models.PyLBM,
    }[args.model](
        Re=args.reynolds, resolution=args.resolution, obstacle=args.obstacle, theta=args.theta)

    # Run the simulation
    if not args.time:
        gui.run(model)
    else:
        while (model.t < args.time):
            model.step()

        print(model.drag_coefficient, model.lift_coefficient)
