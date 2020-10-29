"""Multithreshold Otsu."""
from skimage import data

import numpy as np
import matplotlib.pyplot as plt

img = data.camera()
rows, columns = img.shape

histogram = np.zeros(256)

for row in range(rows):
    for column in range(columns):
        histogram[img[row, column]] += 1

probability = histogram / (rows * columns)

plt.figure(1)
plt.plot()

w0 = []
w1 = []
m0 = []
m1 = []

a = 0.000000001
c = 0

for i in range(256):
    a += probability[i]
    c += i * probability[i]
    w0.append(a)
    m0.append(c / a)
    b = 0.000000001
    d = 0
    for j in range(i + 1, 256):
        b += probability[j]
        d += j * probability[j]
    w1.append(b)
    m1.append(d / b)

w0 = np.array(w0)
w1 = np.array(w1)
m0 = np.array(m0)
m1 = np.array(m1)

threshold = w0 * w1 * ((m1 - m0)**2)
u = np.argmax(threshold)
print(u)

plt.figure(2)
plt.plot(w0)
plt.plot(w1)
plt.legend(['w0', 'w1'])
plt.show()
