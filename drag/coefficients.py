#!/usr/bin/env python3
"""Run this script to calculate the drag and life coefficient of 2d objects in
a 1x1 pipe.
"""
import numpy as np

import code.args as args
import code.models as models


if __name__ == "__main__":
    # Handle command line arguments
    parser = args.ModelArgParser(
        description="Program that tries to calculate the drag and lift "
        "coefficient of 2d objects in a 1x1 pipe.")


    args, model = parser.parse_args()

    print(model.drag_coefficient, model.lift_coefficient)
