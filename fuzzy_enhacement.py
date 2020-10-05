"""Fuzzy Image Enhancement."""
from skimage import data
import skfuzzy as fuzz

import numpy as np
import matplotlib.pyplot as plt

img = data.camera()
rows, columns = img.shape

# Set the range for the fuzzyfication
pixels = np.linspace(0, 255, 256)

# Fuzzyfication of gray levels
dark = fuzz.zmf(pixels, 25, 130)
light = fuzz.smf(pixels, 130, 230)
gray = fuzz.gbellmf(pixels, 55, 3, 128)

# Singletons values
s1 = 30
s2 = 120
s3 = 220

new_gray = np.zeros(256)

# Defuzzyfication process
for i in range(256):
    new_gray[i] = (dark[i] * s1) + (gray[i] * s2) + (light[i] * s3)
    new_gray[i] /= (dark[i] + gray[i] + light[i])

fhe = np.zeros((rows, columns))

# Rule-based enhancement
for row in range(rows):
    for column in range(columns):
        value = img[row, column]
        fhe[row, column] = new_gray[value]

plt.figure(1)
plt.plot(dark)
plt.plot(light)
plt.plot(gray)

plt.figure(2)
plt.imshow(fhe, cmap='gray')

plt.show()
