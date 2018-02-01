# -*- coding: utf-8 -*-
"""This module defines model argument parsing tools, to simplify and unify
the scripts command line argument integration.
"""

import argparse

import pcs_aero.models as models
import pcs_aero.obstacles as obstacles

# To not overflow the help interface of arugment parser only the main obstacle
# names are # shown as possible options
_onames = [l[0] for l in obstacles.names_lists]
_onames_map = {n: l[0] for l in obstacles.names_lists for n in l}


# Defines the type for argument parser
def _onames_type(string):
    string = str.lower(string)
    try:
        return _onames_map[string]
    except KeyError:
        raise TypeError(string, 'not a known obstacle name')


class ModelArgParser(argparse.ArgumentParser):
    """An argument parser that always parsed the arguments required to identify
    a model.

    In essence this class adds some flags that will always be passed.

    Flags:
        '-m', '--model': Model class name.
        '-r', '--resolution': Resolution of model.
        '-o', '--obstacle': Obstacle name.
        '-t', '--theta': Obstacle angle of attack.
        '-u', '--stream_velocity': Uin of model.
        '-s', '--size': Size of obstacle.
        '-T', '--time': The model will be ran at least so far in simulation
            time before being returned from parse_args.
        '-l', '--load': Ignore most other arguments and just load a model from
            some file.

    Note: The only other arugment that does something when the load flag is 
        parsed is the time argument.
    """

    def __init__(self, *args, **kwargs):
        super(ModelArgParser, self).__init__(*args, **kwargs)
        self.add_argument(
            "-m",
            "--model",
            help=
            "The model to use. All models use lattice boltzmann methods, the "
            "difference lies in the implementation. [srt] single relaxation time. "
            "[mrt] multiple relaxation times [pylbm] mrt model that wraps pyLBM",
            type=str.lower,
            choices=['srt', 'mrt', 'trt'],
            default='trt')

        self.add_argument(
            "-r",
            "--resolution",
            help='Number of cells per 1 characteristic length',
            type=int,
            default=92)

        self.add_argument(
            "-R",
            "--reynolds",
            help='Reynolds number',
            type=float,
            default=17.5)

        self.add_argument(
            "-o",
            "--obstacle",
            help='The obstacle to calculate the drag coefficient of.',
            type=_onames_type,
            choices=_onames,
            default=_onames[0])

        self.add_argument(
            "-t",
            "--theta",
            type=float,
            help="Angle of approach in degrees",
            default=0)

        self.add_argument(
            '-u',
            '--stream_velocity',
            type=float,
            help="Base stream lattice velocity, reasonable values are in the "
            "interval [0.01, 0.15]",
            default=0.1)

        self.add_argument(
            '-s',
            '--size',
            type=float,
            help='Reference size (height) of obstacle in characteristic units.',
            default=1 / 8)

        self.add_argument(
            '-T',
            '--time',
            type=float,
            help="Minimum time the simulation has to be ran.",
            default=0)

        self.add_argument(
            '-l',
            '--load',
            type=str,
            help='Load model from file using pickle. This overrides all other '
            'settings.')

    def parse_args(self, *args, **kwargs):
        """Parse the command line arguments.

        Returns: (args, model)
        """
        args = super(ModelArgParser, self).parse_args(*args, **kwargs)

        if args.load:
            import pickle
            with open(args.load, 'rb') as f:
                model = pickle.load(f)

        else:
            # Create the model
            model = {
                "srt": models.SRT,
                "mrt": models.MRT,
                "trt": models.TRT,
            }[args.model](
                # }[args.model](
                Re=args.reynolds,
                resolution=args.resolution,
                obstacle=args.obstacle,
                theta=args.theta,
                Uin=args.stream_velocity,
                size=args.size)

        while model.time < args.time:
            model.step()

        return args, model