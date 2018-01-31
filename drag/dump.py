#!/usr/bin/env python3
import pickle

import code.args as args
import code.models as models


if __name__ == "__main__":
    # Handle command line arguments
    parser = args.ModelArgParser(description='Run a simulation and dump the '
        'object to a file.')

    args, model = parser.parse_args()

    fname = f'R{model.Re}_N{model.N}_o{model.obstacle_name}_t{model.obstacle_theta}_u{model.Uin}_s{model.obstacle_size}.model'

    with open(fname, 'wb') as f:
        pickle.dump(model, f)