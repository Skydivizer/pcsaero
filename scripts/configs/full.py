import numpy as np

from pcs_aero.experiment import ExperimentGroup


# Convenience
_all_shapes = ['circle', 'rect', 'halfcircle', 'triangle', 'moon', 'bullet']

### Actual settings
# Script to use.
program = "scripts/coef.py"

# Experiments to run
experiments = [
    # Note that rotating the circle is pretty useless, since the simulation is
    # deterministic nothing will change. For efficiency one might remove the
    # circle from this group and just simulate it manually once.
    ExperimentGroup("scripts/results/theta.txt", {
        # 'shape': _all_shapes[1:],
        'shape': _all_shapes,
        'theta': range(0, 181, 5),
        'time': [60],
        'Re': [105],
    }),

    ExperimentGroup('scripts/results/reynolds.txt', {
        'shape': _all_shapes,
        'Re': range(55, 106, 2),
        'time': [60],
    }),

    ExperimentGroup('scripts/results/velocity.txt', {
        'shape': _all_shapes,
        'Uin': np.arange(1, 16) / 100,
        'time': [180],
    }),

    ExperimentGroup('scripts/results/size.txt', {
        'shape': _all_shapes,
        'size': np.arange(1, 24) / 64,
        'time': [40],
    }),

    ExperimentGroup('scripts/results/circle_reynolds.txt', {
        'shape': ['circle'],
        'Re': range(10, 1001, 10),
        'time': [60],
    }),
]