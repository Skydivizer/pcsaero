#!/usr/bin/env python3
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

import pcs_aero.args as args
import pcs_aero.models as models


def make_image(data, odata, cm=cm.coolwarm):
    min_v = np.min(data)
    data = (data - min_v)

    max_v = np.max(data)
    data = data / max_v

    data[[0, 1, 2]] = data[[2, 1, 0]]

    image = (cm(data) * 255).astype(np.uint8)
    
    image[odata, :] = [0, 0, 0, 255]

    return image[::-1]


if __name__ == "__main__":
    # Handle command line arguments
    parser = args.ModelArgParser(description="Create a screenshot of some "
                                 "simulation.")

    parser.add_argument('file_name', type=str, help='File name to save as.')

    args, model = parser.parse_args()

    o = model.obstacle_mask.T
    v = model.velocity.T

    vi = make_image(v, o)

    img = Image.fromarray(vi, 'RGBA')
    img.save(args.file_name)
