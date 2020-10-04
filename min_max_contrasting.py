"""Min Max Contrasting of an Image."""
from skimage import data

import numpy as np
import matplotlib.pyplot as plt


def get_histogram(img):
    """Return the histogram of a grayscale image.

    img: numpy array [n, m, 1]
    returns: numpy array [256, 1, 1]
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


def min_max_contrasting(img):
    """Return the image after a min max contrasting.

    img: numpy array [n, m, 1]
    return: output, numpy array to plot as image [n, m, 1]
    """
    rows, columns = img.shape

    output = np.zeros((rows, columns), dtype='uint8')

    # Loop over the image and apply Min-Max Contrasting
    img_min = np.min(img)
    img_max = np.max(img)

    for row in range(rows):
        for column in range(columns):
            output[row, column] = 255 * ((img[row, column] - img_min) / (img_max - img_min))

    return output


# Original data

img = data.camera()
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
axs1[1].set_title('HE Image')

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
