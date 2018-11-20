import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('/Users/huhu/Documents/cell/1.tif', 0)
#cv2.imshow('img', img)
#cv2.waitKey(0)

print(img.shape)

#img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#cv2.imshow('img_gray', img_gray)
#cv2.waitKey(0)

hist = cv2.calcHist([img], [0], None, [256], [0,255])
print(hist)
#cv2.imshow('hist', hist)
#cv2.waitKey(0)

plt.figure()
plt.plot(hist)
plt.xlim([0,255])
plt.show()

ret, thre = cv2.threshold(img, 90, 255, cv2.THRESH_BINARY)
print(thre.shape)
plt.figure()
plt.imshow(thre, 'gray')
plt.show()