#!/usr/bin/env python3

import code.args as args
import code.models as models
import code.gui as gui


if __name__ == "__main__":
    # Handle command line arguments
    parser = args.ModelArgParser(description='Run a simulation in an opengl '
        'window')

    args, model = parser.parse_args()

    gui.run(model)
