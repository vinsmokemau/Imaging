"""Min Max Contrasting of an Image."""
from skimage import data, color, io

import numpy as np
import matplotlib.pyplot as plt


def get_histogram(image):
    """Return the histogram of a grayscale image.

    image: numpy array [n, m, 1]
    returns: numpy array [256, 1, 1]
    """
    rows, columns = image.shape

    histogram = np.zeros(256)

    for row in range(rows):
        for column in range(columns):
            position = image[row, column]
            histogram[position] += 1

    return histogram


def get_cumulative_distribution(image):
    """Return the cumulative distributuion of a grayscale image.

    image: numpy array [n, m, 1]
    return: numpy array to plot as graph [256, 1, 1]
    """
    rows, columns = image.shape

    probability = np.zeros(256)
    probability = histogram / (rows * columns)

    cumulative_dist = probability.cumsum()

    return cumulative_dist


def min_max_contrasting(image):
    """Return the image after a min max contrasting.

    image: numpy array [n, m, 1]
    return: output, numpy array to plot as image [n, m, 1]
    """
    rows, columns = image.shape

    output = np.zeros((rows, columns), dtype='uint8')

    # Loop over the image and apply Min-Max Contrasting
    image_min = np.min(image) * 1.1
    image_max = np.max(image) // 1.1

    for row in range(rows):
        for column in range(columns):
            output[row, column] = 255 * ((image[row, column] - image_min) / (image_max - image_min))

    return output


# Original data

img = np.uint8(color.rgb2gray(io.imread('img/img1.jpeg')) * 255)
histogram = get_histogram(img)
cumulative_dist = get_cumulative_distribution(img)
# Normalize cumulative distribution
cumulative_dist *= (histogram.max() / cumulative_dist.max())

# Min Max Contrasting data

mmc_img = min_max_contrasting(img)
mmc_histogram = get_histogram(mmc_img)
mmc_cumulative_dist = get_cumulative_distribution(mmc_img)
# Normalize cumulative distribution
mmc_cumulative_dist *= (mmc_histogram.max() / mmc_cumulative_dist.max())

# Plot images

fig1, axs1 = plt.subplots(1, 2)
# Plot the original image
axs1[0].imshow(img, cmap="gray")
axs1[0].set_title('Original Image')
# Plot the image after histogram equalization
axs1[1].imshow(mmc_img, cmap="gray")
axs1[1].set_title('MMC Image')

# Plot histograms and cumulative distributions

fig2, axs2 = plt.subplots(1, 2)
# Plot the histogram and cumulative distribution of the original image
axs2[0].plot(histogram, color='r')
axs2[0].plot(cumulative_dist, color='b')
axs2[0].set_title('Original Image')
axs2[0].legend(('histogram', 'cdf'), loc='upper left')
# Plot the image after histogram equalization
axs2[1].plot(mmc_histogram, color='r')
axs2[1].plot(mmc_cumulative_dist, color='b')
axs2[1].set_title('MMC Image')
axs2[1].legend(('histogram', 'cdf'), loc='upper left')

plt.show()
