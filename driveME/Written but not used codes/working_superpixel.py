# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 14:54:46 2019

@author: Kazım
"""

import matplotlib.pyplot as plt
import numpy as np

from skimage.data import astronaut
from skimage.color import rgb2gray
from skimage.filters import sobel, scharr
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
import cv2
import skimage
import time
from scipy.spatial import distance as dist
width = 540
height = 960
scaling_ratio = 1
new_width = scaling_ratio*width
new_width = (int)(new_width)
new_height = scaling_ratio*height
new_height = (int)(new_height)
img = cv2.imread('D:\\ground_detection\\normal2.jpg')

img_smooth = cv2.bilateralFilter(img, 5, 20, 20)
img_smooth = cv2.bilateralFilter(img_smooth, 5, 20, 20)
img_smooth = cv2.bilateralFilter(img_smooth, 5, 20, 20)


img_smooth = cv2.cvtColor(img_smooth, cv2.COLOR_BGR2RGB) 
gray = cv2.cvtColor(img_smooth, cv2.COLOR_BGR2GRAY) 

hsv = cv2.cvtColor(img_smooth, cv2.COLOR_BGR2HSV) 
hsv = img_smooth 
img_smooth = cv2.resize(img_smooth,(new_height,new_width))  
hsv_small = cv2.cvtColor(img_smooth, cv2.COLOR_BGR2LAB) 
hsv_small = img_smooth 
img_smooth = cv2.cvtColor(img_smooth, cv2.COLOR_BGR2RGB) 
img_smooth = img_as_float(img_smooth)



start_time = time.time()
segments_fz = felzenszwalb(img_smooth, scale=1000, sigma=0.8, min_size=10)
elapsed_time = time.time() - start_time
print(elapsed_time)

start_time = time.time()
segments_slic = slic(img_smooth, n_segments=200, compactness=5, sigma=0.5)
elapsed_time = time.time() - start_time
print(elapsed_time)

start_time = time.time()
segments_quick = quickshift(img_smooth, kernel_size=3, max_dist=6, ratio=0.5)
elapsed_time = time.time() - start_time
print(elapsed_time)

start_time = time.time()
gradient = sobel(rgb2gray(img_smooth))
segments_watershed = watershed(gradient, markers=250, compactness=0.0005)
elapsed_time = time.time() - start_time
print(elapsed_time)


print("Felzenszwalb number of segments: {}".format(len(np.unique(segments_fz))))
print('SLIC number of segments: {}'.format(len(np.unique(segments_slic))))
print('Quickshift number of segments: {}'.format(len(np.unique(segments_quick))))
print('Watershed number of segments: {}'.format(len(np.unique(segments_watershed))))

fig, ax = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)

ax[0, 0].imshow(mark_boundaries(img_smooth, segments_fz))
ax[0, 0].set_title("Felzenszwalbs's method")
ax[0, 1].imshow(mark_boundaries(img_smooth, segments_slic))
ax[0, 1].set_title('SLIC')
ax[1, 0].imshow(mark_boundaries(img_smooth, segments_quick))
ax[1, 0].set_title('Quickshift')
ax[1, 1].imshow(mark_boundaries(img_smooth, segments_watershed))
ax[1, 1].set_title('Compact watershed')

for a in ax.ravel():
    a.set_axis_off()

plt.tight_layout()
plt.show()





start_time = time.time()
segments = felzenszwalb(img_smooth, scale=100, sigma=0.5, min_size=50)
#segments = slic(img_smooth, n_segments=250, compactness=10, sigma=1.0)
threshold = np.zeros((new_width,new_height),dtype=np.uint8)
(means, stds) = cv2.meanStdDev(hsv[440:470,310:330,:])
means[2,0] = means[2,0]
means[1,0] = means[1,0]
means[0,0] = means[0,0]*0.1
stds[0,0] = stds[0,0]*0.1
features = np.concatenate([means, stds]).flatten()
for i in range(len(np.unique(segments))):
    mask_label = 255*(segments==i)
    mask_label=mask_label.astype(np.uint8)
    (means, stds) = cv2.meanStdDev(hsv_small,mask=mask_label)
    means[2,0] = means[2,0]
    means[1,0] = means[1,0]
    means[0,0] = means[0,0]*0.1
    stds[0,0] = stds[0,0]*0.1
    label_feature = np.concatenate([means, stds]).flatten()
    d = dist.euclidean(features, label_feature)
    #if d>120:
    #    mask_label = mask_label*0
    mask_label = d*(mask_label==255)
    mask_label = mask_label.astype(np.uint8)
    threshold = cv2.add(threshold,mask_label)


threshold = cv2.bitwise_not(threshold)
threshold = (threshold.astype(float)-np.min(threshold))*255/(np.max(threshold)-np.min(threshold))
out=cv2.convertScaleAbs(threshold.astype(np.uint8))
out = cv2.bilateralFilter(out, 5, 20, 20)
#out = cv2.bilateralFilter(out, 10, 75, 75)
elapsed_time = time.time() - start_time
out = cv2.resize(out,(640,480))
cv2.imshow('occupancy', out)
ret,thresh1 = cv2.threshold(out,220,255,cv2.THRESH_BINARY)
kernel = np.ones((3,3),np.uint8)
thresh1 = cv2.dilate(thresh1,kernel,iterations = 2)
thresh1 = cv2.erode(thresh1,kernel,iterations = 2)

cv2.imshow('occupancy_thres', thresh1)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('D:\\ground_detection\\thresh.png',thresh1)