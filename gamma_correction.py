"""Histogram Equalization of an Image."""
from skimage import data, color, io

import numpy as np
import matplotlib.pyplot as plt


def get_histogram(img):
    """Return the histogram of a grayscale image.

    img: numpy array [n, m, 1]
    return: numpy array [256, 1, 1]
    """
    rows, columns = img.shape

    histogram = np.zeros(256)

    for row in range(rows):
        for column in range(columns):
            position = img[row, column]
            histogram[position] += 1

    return histogram


def get_cumulative_distribution(img):
    """Return the cumulative distributuion of a grayscale image.

    img: numpy array [n, m, 1]
    return: numpy array to plot as graph [256, 1, 1]
    """
    rows, columns = img.shape

    probability = np.zeros(256)
    probability = histogram / (rows * columns)

    cumulative_dist = probability.cumsum()

    return cumulative_dist


def gamma_correction(img, gamma):
    """Return the image after a gamma correction.

    img: numpy array [n, m, 1]
    return: numpy array to plot as graph [256, 1, 1]
    """
    rows, columns = img.shape

    output = np.zeros((rows, columns), dtype='uint8')

    for row in range(rows):
        for column in range(columns):
            output[row, column] = (img[row, column] ** gamma) * 255

    return output


# Original data

img = np.uint8(color.rgb2gray(io.imread('img/img1.jpeg')) * 255)
histogram = get_histogram(img)
cumulative_dist = get_cumulative_distribution(img)
# Normalize cumulative distribution
cumulative_dist *= (histogram.max() / cumulative_dist.max())

# Gamma Correction data

gc_img = gamma_correction(img, 0.05)
gc_histogram = get_histogram(gc_img)
gc_cumulative_dist = get_cumulative_distribution(gc_img)
# Normalize cumulative distribution
gc_cumulative_dist *= (gc_histogram.max() / gc_cumulative_dist.max())

# Plot images

fig1, axs1 = plt.subplots(1, 2)
# Plot the original image
axs1[0].imshow(img, cmap="gray")
axs1[0].set_title('Original Image')
# Plot the image after histogram equalization
axs1[1].imshow(gc_img, cmap="gray")
axs1[1].set_title('GC Image')

# Plot histograms and cumulative distributions

fig2, axs2 = plt.subplots(1, 2)
# Plot the histogram and cumulative distribution of the original image
axs2[0].plot(histogram, color='r')
axs2[0].plot(cumulative_dist, color='b')
axs2[0].set_title('Original Image')
axs2[0].legend(('histogram', 'cdf'), loc='upper left')
# Plot the image after histogram equalization
axs2[1].plot(gc_histogram, color='r')
axs2[1].plot(gc_cumulative_dist, color='b')
axs2[1].set_title('GC Image')
axs2[1].legend(('histogram', 'cdf'), loc='upper left')

plt.show()
