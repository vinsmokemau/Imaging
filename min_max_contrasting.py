"""Get the Histogram of an Image."""
from skimage import data

import numpy as np
import matplotlib.pyplot as plt


def get_histogram(img):
    """Return the histogram of a grayscale image.

    img: numpy array [n, m, 1]
    returns: numpy array [256, 1, 1]
    """
    [rows, columns] = img.shape

    histogram = np.zeros(256)

    for row in range(rows):
        for column in range(columns):
            position = img[row, column]
            histogram[position] += 1

    return histogram


img = data.camera()

histogram = get_histogram(img)

rows, columns = img.shape

probability = np.zeros(256)
probability = histogram / (rows * columns)

cumulative_dist = probability.cumsum()

output = np.zeros((rows, columns), dtype='uint8')

# Loop over the image and apply Min-Max Contrasting
img_min = np.min(img)
img_max = np.max(img)

for row in range(rows):
    for column in range(columns):
        output[row, column] = 255 * (img[row, column] - img_min) / (img_max - img_min)

plt.figure(1)
# Plot the original image
plt.imshow(img, cmap="gray")

plt.figure(2)
# Plot the histogram of the original image
plt.plot(histogram)

plt.figure(3)
# Plot the cumulative distribution function of the original image
plt.plot(cumulative_dist)

plt.figure(4)
# Plot the image after histogram equalization
plt.imshow(output, cmap="gray")

plt.show()
