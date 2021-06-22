import cv2
import numpy as np
import matplotlib.pyplot as plt

img1 = cv2.imread('D:/phoenix/pic-1.jpg')
print(img1.dtype)
print(img1.shape)
plt.imshow(img1)
#plt.show()
plt.axis('off')
plt.tight_layout()
plt.savefig('D:/phoenix/pic-1.jpg')
print(img1.dtype)
print(img1.shape)
img2 = cv2.imread('D:/phoenix/pic-1.jpg')

#img2 = img2.resize(210, 160, 3)
print(img2.dtype)
print(img2.shape)

img_mix = cv2.addWeighted(img1, 0.6, img2, 0.4, 0)

#cv2.imshow('img1', img1)
#cv2.imshow('img2', img2)
#cv2.imshow('img_mix', img_mix)
