import numpy as np

from pcs_aero.experiment import ExperimentGroup

# Convenience
_all_shapes = ['rect', 'halfcircle', 'triangle']

# Script to use.
program = "scripts/coef.py"

experiments = [
    ExperimentGroup("scripts/results/theta.txt", {
        'shape': _all_shapes,
        'theta': range(0, 181, 30),
        'time': [20],
        'Re': [105],
        'r': [64],
    }),

    ExperimentGroup('scripts/results/reynolds.txt', {
        'shape': _all_shapes,
        'Re': range(55, 106, 10),
        'time': [20],
        'r': [64],
    }),

    ExperimentGroup('scripts/results/velocity.txt', {
        'shape': _all_shapes,
        'Uin': np.arange(4, 16, 2) / 100,
        'time': [30],
        'r': [64],
    }),

    ExperimentGroup('scripts/results/size.txt', {
        'shape': _all_shapes,
        'size': np.arange(5, 24, 3) / 64,
        'time': [20],
        'r': [64],
    }),

    ExperimentGroup('scripts/results/circle_reynolds.txt', {
        'shape': ['circle'],
        'Re': range(10, 1001, 100),
        'time': [40],
        'r': [64],
    }),
]