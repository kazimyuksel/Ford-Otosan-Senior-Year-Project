# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 20:44:30 2019

@author: Kazım
"""
import matplotlib.pyplot as plt
import numpy as np
import cv2

kernel = np.ones((3,3), np.uint8) 
image = cv2.imread("D:\\FordOtosan\\occupancy.PNG",0)
#image = cv2.imread("D:\\_ravel.PNG",0)
#threshold image to get binary image
_,source_image = cv2.threshold(image,120,255,cv2.THRESH_BINARY)
img_erosion = cv2.erode(source_image, kernel, iterations=1) 
dist = cv2.distanceTransform(img_erosion, cv2.DIST_L2, 5)
pdf = dist / np.sum(dist);
#dist = np.power(dist,0.9)
dist_1d = pdf.ravel()
# Normalize the distance image for range = {0.0, 1.0}
# so we can visualize and threshold it
cv2.normalize(dist, dist, 0, 255.0, cv2.NORM_MINMAX)
dist = dist.astype(np.uint8)
cv2.imshow('Distance Transform Image', dist)
#dist = cv2.GaussianBlur(dist,(7,7),5)
#cv2.imshow('Distance Transform Image BLUR', dist)
cv2.imshow('eroded', img_erosion)
a = np.cumsum(dist_1d)
u, indices = np.unique(a, return_index=True)
u = u[1:]
indices = indices[1:]
#elems = np.arange(0, a.shape[0], 1)
#plt.scatter(elems,a)
#plt.show()
#
height =1000
width =1000
circle_img = np.zeros((height,width), np.uint8)
cv2.circle(circle_img,(int(width/2),int(height/2)),500,255,thickness=-1)
cv2.imshow('eroded1', circle_img)
cv2.waitKey(0)
#destroy all windows
cv2.destroyAllWindows()


rand_point=np.random.random_sample()
def find_nearest(array_index,array_cdf, value):
    array_cdf = np.asarray(array_cdf)
    idx = (np.abs(array_cdf - value)).argmin()
    return array_index[idx]


initial_array = np.zeros(dist.shape)
while True:
    rand_point=np.random.random_sample()
    #print(find_nearest(indices,u, rand_point))
    nearest_index = find_nearest(indices,u, rand_point)
    x = np.mod(nearest_index,dist.shape[1])
    y = int(nearest_index/dist.shape[1])
    initial_array[y,x] = initial_array[y,x] + 100
    if initial_array[y,x] > 255:
        initial_array[y,x] = 255
    cv2.imshow('eroded', initial_array.astype(np.uint8))
    if cv2.waitKey(27) & 0xFF == ord('q') :
        break
cv2.waitKey(0)
cv2.destroyAllWindows()


