#!/usr/bin/env python3
import numpy as np

import pcs_aero.args as args
import pcs_aero.models as models

if __name__ == "__main__":
    parser = args.ModelArgParser(
        description="Prints the drag and lift coefficient of some model.")

    args, model = parser.parse_args()

    print(model.drag_coefficient, model.lift_coefficient)
