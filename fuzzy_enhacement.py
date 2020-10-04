"""Get the Histogram of an Image."""
from skimage import data

import numpy as np
import matplotlib.pyplot as plt

dark = np.zeros(256)

for pixel in range(256):
    if pixel < 35:
        dark = 1
    elif pixel >= 35 and pixel <= 100:
        dark = (100 - pixel) / 65  # (100-pixel) / (100-35)
    else:
        dark = 0
