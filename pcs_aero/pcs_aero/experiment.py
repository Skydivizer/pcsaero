# -*- coding: utf-8 -*-
"""This module can be used to quickly generate a product of multiple setting
ranges.

This could be for instance in a parameter sweep, or just running loads of
simulations.
"""
import itertools


class Experiment(object):
    """This class defines a single experiment.

    It can be used to generate the command line arguments associated with an
    ModelArgParser.
    """

    def __init__(
            self,
            shape='circle',
            model='trt',
            theta=0,
            r=92,
            Re=100,
            time=60,
            Uin=0.1,
            size=1 / 8,
            id_format="{model} {shape} {theta} {r} {Re} {time} {Uin} {size}"):
        """Initialize Experiment class.

        Arguments:
            shape (str): Name of the obstacle.
            model (str): Model name.
            theta (float): Model obstacle theta.
            r (int): Model resolution.
            Re (float): Model reynolds number.
            time (float): Time to run model.
            Uin (float): Model Uin.
            size (float): Model obstacle size.
            id_format (str): String format to use as identifier for this
                experiment, e.g. "{shape} {theta}".
        """

        self.model = model
        self.shape = shape
        self.r = r
        self.Re = Re
        self.time = time
        self.theta = theta
        self.Uin = Uin
        self.size = size

        self.id = id_format.format(
            model=self.model,
            shape=self.shape,
            r=self.r,
            Re=self.Re,
            time=self.time,
            theta=self.theta,
            Uin=self.Uin,
            size=self.size)

    def generate_args(self):
        """Generate argument flags associated with this models.
        
        Returns:
            str[8]: List of flags.
        """
        return [
            "-m{}".format(self.model),
            "-r{}".format(self.r),
            "-R{}".format(self.Re),
            "-T{}".format(self.time),
            "-o{}".format(self.shape),
            "-t{}".format(self.theta),
            "-u{}".format(self.Uin),
            "-s{}".format(self.size),
        ]


class ExperimentGroup():
    """Group of experiments."""

    def __init__(self, name, var):
        """Create the group.

        Arguments:
            name (str): Experiment name.
            var (dict(str, list)): Settings to iterate over.

        Note:
            All settings need to be iterable, even if it is a constant, e.g.
            `'theta': 0` is not allowed but `'theta': [0]` is.
        """
        self.name = name

        self.experiments = []

        self.ids = [key for key in var.keys() if len(var[key]) > 1]

        id_string = ""
        for i in self.ids:
            id_string += f"{{{i}}} "

        for values in itertools.product(*var.values()):
            args = dict(zip(var.keys(), values))
            self.experiments.append(Experiment(**args, id_format=id_string))
