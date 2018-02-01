#!/usr/bin/env python3
import pickle

import pcs_aero.args as args
import pcs_aero.models as models

if __name__ == "__main__":
    parser = args.ModelArgParser(description='Run a simulation and dump the '
                                 'object to a file.')

    parser.add_argument('file_name', type=str, help='File name to save as.')

    args, model = parser.parse_args()

    with open(args.file_name, 'wb') as f:
        pickle.dump(model, f)