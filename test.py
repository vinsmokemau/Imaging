"""."""
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('img/chest.jpg')
cv2.imshow("Images", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

gray_img_eqhist = cv2.equalizeHist(gray_img)
hist = cv2.calcHist(gray_img_eqhist, [0], None, [256], [0, 256])
cv2.imshow("Images", gray_img_eqhist)
cv2.imwrite('img/chest_eqhist.jpg', gray_img_eqhist)
cv2.waitKey(0)
cv2.destroyAllWindows()

clahe = cv2.createCLAHE(clipLimit=40)
gray_img_clahe = clahe.apply(gray_img_eqhist)
cv2.imshow("Images", gray_img_clahe)
cv2.imwrite('img/chest_clahe.jpg', gray_img_clahe)
cv2.waitKey(0)
cv2.destroyAllWindows()
