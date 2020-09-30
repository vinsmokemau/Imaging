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


def histogram_equalization(img):
    """
    Return the graph and image of the histogram equalization.

    img: numpy array [n, m, 1]
    return: graph, numpy array to plot as graph [256, 1, 1]
            output, numpy array to plot as image [n, m, 1]
    """
    [rows, columns] = img.shape

    histogram = get_histogram(img)

    probability = np.zeros(256)
    probability = histogram / (rows * columns)

    hist_equalization = probability.cumsum()

    output = np.zeros((rows, columns))

    for row in range(rows):
        for column in range(columns):
            position = img[row, column]
            output[row, column] = np.uint8(hist_equalization[position] * 255)

    return hist_equalization, output


img = data.camera()

histogram = get_histogram(img)
he_graph, he_img = histogram_equalization(img)

plt.figure(1)
# Plot image
plt.imshow(img, cmap="gray")

plt.figure(2)
# Plot the img histogram
plt.plot(histogram)

plt.figure(3)
# Plot the histogram equalization
plt.plot(he_graph)

plt.figure(4)
# Plot the image after hostogram equalization
plt.imshow(he_img, cmap="gray")

plt.show()
