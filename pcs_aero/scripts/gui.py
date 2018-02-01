#!/usr/bin/env python3

import pcs_aero.args as args
import pcs_aero.models as models
import pcs_aero.gui as gui

if __name__ == "__main__":
    parser = args.ModelArgParser(description='Run a simulation in an opengl '
                                 'window')

    args, model = parser.parse_args()

    gui.run(model)
