#!/usr/bin/env python3
import numpy as np

import pcs_aero.args as args
import pcs_aero.models as models


if __name__ == "__main__":
    parser = args.ModelArgParser(
        description="Program that tries to calculate the drag and lift "
        "coefficient of 2d objects in a 1x1 pipe.")


    args, model = parser.parse_args()

    print(model.drag_coefficient, model.lift_coefficient)
