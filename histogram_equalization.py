"""Histogram Equalization of an Image."""
from skimage import data

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


def histogram_equalization(img):
    """Return the image after a histogram equalization.

    img: numpy array [n, m, 1]
    return: output, numpy array to plot as image [n, m, 1]
    """
    rows, columns = img.shape

    cumulative_dist = get_cumulative_distribution(img)

    output = np.zeros((rows, columns), dtype='uint8')

    for row in range(rows):
        for column in range(columns):
            position = img[row, column]
            output[row, column] = cumulative_dist[position] * 255

    return output


# Original data

img = data.camera()
histogram = get_histogram(img)
cumulative_dist = get_cumulative_distribution(img)
# Normalize cumulative distribution
cumulative_dist *= (histogram.max() / cumulative_dist.max())

# HE data

he_img = histogram_equalization(img)
he_histogram = get_histogram(he_img)
he_cumulative_dist = get_cumulative_distribution(he_img)
# Normalize cumulative distribution
he_cumulative_dist *= (he_histogram.max() / he_cumulative_dist.max())

# Plot images

fig1, axs1 = plt.subplots(1, 2)
# Plot the original image
axs1[0].imshow(img, cmap="gray")
axs1[0].set_title('Original Image')
# Plot the image after histogram equalization
axs1[1].imshow(he_img, cmap="gray")
axs1[1].set_title('HE Image')

# Plot histograms and cumulative distributions

fig2, axs2 = plt.subplots(1, 2)
# Plot the histogram and cumulative distribution of the original image
axs2[0].plot(histogram, color='r')
axs2[0].plot(cumulative_dist, color='b')
axs2[0].set_title('Original Image')
axs2[0].legend(('histogram', 'cdf'), loc='upper left')
# Plot the image after histogram equalization
axs2[1].plot(he_histogram, color='r')
axs2[1].plot(he_cumulative_dist, color='b')
axs2[1].set_title('HE Image')
axs2[1].legend(('histogram', 'cdf'), loc='upper left')

plt.show()
