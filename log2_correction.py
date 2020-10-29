"""Logarithm Base 2 Correction of an Image."""
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


def log2_correction(img):
    """Return the image after a gamma correction.

    img: numpy array [n, m, 1]
    return: numpy array to plot as graph [256, 1, 1]
    """
    rows, columns = img.shape

    output = np.zeros((rows, columns), dtype='uint8')

    for row in range(rows):
        for column in range(columns):
            output[row, column] = np.log2(img[row, column] + 1) * 255

    return output


# Original data

img = np.uint8(color.rgb2gray(io.imread('img/img1.jpeg')) * 255)
histogram = get_histogram(img)
cumulative_dist = get_cumulative_distribution(img)
# Normalize cumulative distribution
cumulative_dist *= (histogram.max() / cumulative_dist.max())

# Gamma Correction data

log2_img = log2_correction(img)
log2_histogram = get_histogram(log2_img)
log2_cumulative_dist = get_cumulative_distribution(log2_img)
# Normalize cumulative distribution
log2_cumulative_dist *= (log2_histogram.max() / log2_cumulative_dist.max())

# Plot images

fig1, axs1 = plt.subplots(1, 2)
# Plot the original image
axs1[0].imshow(img, cmap="gray")
axs1[0].set_title('Original Image')
# Plot the image after histogram equalization
axs1[1].imshow(log2_img, cmap="gray")
axs1[1].set_title('Log 2 Image')

# Plot histograms and cumulative distributions

fig2, axs2 = plt.subplots(1, 2)
# Plot the histogram and cumulative distribution of the original image
axs2[0].plot(histogram, color='r')
axs2[0].plot(cumulative_dist, color='b')
axs2[0].set_title('Original Image')
axs2[0].legend(('histogram', 'cdf'), loc='upper left')
# Plot the image after histogram equalization
axs2[1].plot(log2_histogram, color='r')
axs2[1].plot(log2_cumulative_dist, color='b')
axs2[1].set_title('Log 2 Image')
axs2[1].legend(('histogram', 'cdf'), loc='upper left')

plt.show()
