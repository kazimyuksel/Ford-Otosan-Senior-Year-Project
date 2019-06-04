# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 17:45:57 2019

@author: Kazım
"""

from __future__ import division

import os
import sys
import math
import datetime
import time

import cv2
assert cv2.__version__[0] >= '3', 'The program requires OpenCV version greater than >= 3.0.0'
from cv2 import aruco
import numpy as np
from colorama import Fore, Style
from skimage.measure import block_reduce as pooling
from skimage import color,io,exposure,transform
from sklearn.cluster import KMeans
from progressbar import ProgressBar, Bar, AdaptiveETA, Percentage, SimpleProgress, Timer
from scipy.spatial import distance as dist
import mahotas


img = cv2.imread('D:\\ground_detection\\3.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.bilateralFilter(gray, 10, 75, 75)


gx = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
gy = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
mag, ang = cv2.cartToPolar(gx, gy)
gX = cv2.convertScaleAbs(gx)
gY = cv2.convertScaleAbs(gy)

scaled_y = 255*(gy-np.min(gy))/(np.max(gy)-np.min(gy))
scaled_x = 255*(gx-np.min(gx))/(np.max(gx)-np.min(gx))
otsu, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY or cv2.THRESH_OTSU)
low_threshold = otsu * 0.5
high_threshold = otsu
canny = cv2.Canny(gray,low_threshold,high_threshold)
added = cv2.addWeighted(gX,0.5,gY,0.5,0)
cv2.imwrite('D:\\ground_detection\\scaled_x.png',scaled_x)

img = cv2.imread('D:\\ground_detection\\5.png')
img_smooth = cv2.bilateralFilter(img, 5, 20, 20)
cv2.imwrite('D:\\ground_detection\\smooth.png',img_smooth)
kernel_size = 5

col_size = 640/kernel_size
row_size = 480/kernel_size

for i in range((int)(col_size)):
    for j in range((int)(row_size)):
        x_start = kernel_size*i
        x_end = kernel_size*(i+1)
        y_start = kernel_size*j
        y_end = kernel_size*(j+1)
  
width = 480
height = 640
scaling_ratio = 1
new_width = scaling_ratio*width
new_width = (int)(new_width)
new_height = scaling_ratio*height
new_height = (int)(new_height)
img = cv2.imread('D:\\ground_detection\\5.png')

img_smooth = cv2.bilateralFilter(img, 5, 20, 20)
img_smooth = cv2.resize(img_smooth,(new_width,new_height))
lab = cv2.cvtColor(img_smooth, cv2.COLOR_BGR2HSV) 
#lab = cv2.cvtColor(img_smooth, cv2.COLOR_BGR2GRAY) 
hsv = cv2.cvtColor(img_smooth, cv2.COLOR_BGR2HSV)    
num_superpixels = 200 # desired number of superpixels
num_iterations = 10  # number of pixel level iterations. The higher, the better quality
prior = 1           # for shape smoothing term. must be [0, 5]
num_levels = 50
num_histogram_bins = 50 # number of histogram bins
height, width,channels= lab.shape

seeds = cv2.ximgproc.createSuperpixelSEEDS(width, height, channels, num_superpixels, num_levels, prior, num_histogram_bins)
start_time = time.time()
seeds.iterate(lab, num_iterations)
num_of_superpixels_result = seeds.getNumberOfSuperpixels()
print('Final number of superpixels: %d' % num_of_superpixels_result)


# retrieve the segmentation result
labels = seeds.getLabels()
threshold = np.zeros((480,640),dtype=np.uint8)
(means, stds) = cv2.meanStdDev(lab[440:470,310:330,:])
means[2,0] = means[2,0]*0.2
features = np.concatenate([means, stds]).flatten()
for i in range(num_of_superpixels_result):
    mask_label = 255*(labels==i)
    mask_label=mask_label.astype(np.uint8)
    (means, stds) = cv2.meanStdDev(lab,mask=mask_label)
    means[2,0] = means[2,0]*0.2
    label_feature = np.concatenate([means, stds]).flatten()
    d = dist.euclidean(features, label_feature)
    #if d>120:
    #    mask_label = mask_label*0
    mask_label = d*(mask_label==255)
    mask_label = mask_label.astype(np.uint8)
    threshold = cv2.add(threshold,mask_label)
elapsed_time = time.time() - start_time
img_smooth_1 = cv2.resize(img_smooth,(640,480))
# draw contour
mask = seeds.getLabelContourMask(False)
cv2.imshow('MaskWindow', mask)
cv2.waitKey(0)
# draw color coded image
color_img = np.zeros((height, width, 3), np.uint8)
color_img[:] = (0, 0, 255)
mask_inv = cv2.bitwise_not(mask)
result_bg = cv2.bitwise_and(img_smooth, img_smooth, mask=mask_inv)
result_fg = cv2.bitwise_and(color_img, color_img, mask=mask)
result = cv2.add(result_bg, result_fg)
cv2.imshow('ColorCodedWindow', result)
threshold = cv2.bitwise_not(threshold)
threshold = (threshold.astype(float)-np.min(threshold))*255/(np.max(threshold)-np.min(threshold))
cv2.waitKey(0)
cv2.imshow('occupancy', threshold.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()



kernel_size = 20
img = cv2.imread('D:\\ground_detection\\5.png')
#img_smooth = cv2.resize(img_smooth,(360,240))
start_time = time.time()
thresh = np.zeros((480,640),dtype=np.uint8)
img_smooth = cv2.bilateralFilter(img, 5, 20, 20)
lab = cv2.cvtColor(img_smooth, cv2.COLOR_BGR2HSV) 
hsv = cv2.cvtColor(img_smooth, cv2.COLOR_BGR2HSV) 
col_size = 640/kernel_size
row_size = 480/kernel_size
(means, stds) = cv2.meanStdDev(lab[440:470,310:330,:])
means[0,0] = means[0,0]
means[2,0] = means[2,0]
features = np.concatenate([means, stds]).flatten()
for i in range((int)(col_size)):
    for j in range((int)(row_size)):
        x_start = kernel_size*i
        x_end = kernel_size*(i+1)
        y_start = kernel_size*j
        y_end = kernel_size*(j+1)
        area = lab[y_start:y_end,x_start:x_end,:]
        (means, stds) = cv2.meanStdDev(area)
        means[0,0] = means[0,0]
        means[2,0] = means[2,0]
        label_feature = np.concatenate([means, stds]).flatten()
        d = dist.euclidean(features, label_feature)
        thresh[y_start:y_end,x_start:x_end] = (int)(d)
elapsed_time = time.time() - start_time
thresh = cv2.bitwise_not(thresh)
thresh = (np.array(thresh).astype(float)-np.min(thresh))*255/(np.max(thresh)-np.min(thresh))
cv2.imshow('occupancy', thresh.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()







