"""."""
from skimage import data, io, color

import numpy as np
import matplotlib.pyplot as plt
import random
from skimage.morphology import disk
from skimage.filter import rank

img = np.unit8(color.rgb2gray(io.imread('img/chest.jpg')) * 255)
plt.figure(1)
plt.subplot(131)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.suptitle('Image Enhancement with AWMF')
 	
row, columns = img.shape
snp = 0.5
amount = 0.9
imr = np.copy(img)
num_salt = np.ceil(amount * img.size * snp)
coords = [np.random.randint(0, i - 1, int(num_salt)) for i in imr.shape]
imr[coords] = 255
num_pepper = np.ceil(amount * imr.size * (1 - snp))
coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in imr.shape]
imr[coords] = 0

plt.subplot(132)
plt.imshow(imr, cmap="gray")
plt.title('S&P noisy Image')


