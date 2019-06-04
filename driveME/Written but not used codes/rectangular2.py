# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 23:49:12 2018

@author: Kazım
"""


import cv2
import numpy as np

img = cv2.imread('D:\\yamuk.jpg',-1)
img = cv2.GaussianBlur(img,(5,5),10)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower_green = np.array([57, 88, 90])
upper_green = np.array([67, 242, 231])
mask = cv2.inRange(hsv, lower_green, upper_green)

cv2.imshow('image', mask)

dst = cv2.cornerHarris(mask,4,3,0.4)

#result is dilated for marking the corners, not important
dst = cv2.dilate(dst,None)

# Threshold for an optimal value, it may vary depending on the image.
img[dst>0.01*dst.max()]=[0,0,255]

cv2.imshow('Corner',img)
cv2.waitKey(0)
cv2.destroyAllWindows()