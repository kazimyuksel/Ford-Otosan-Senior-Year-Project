# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 16:01:57 2019

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
import cupy as cp

img = cv2.imread('D:\\ground_detection\\5.png')
#img_smooth = cv2.resize(img_smooth,(360,240))
img_smooth = cv2.bilateralFilter(img, 5, 20, 20)
lab = cv2.cvtColor(img_smooth, cv2.COLOR_BGR2HSV) 
hsv = cv2.cvtColor(img_smooth, cv2.COLOR_BGR2HSV)    
num_superpixels = 200 # desired number of superpixels
num_iterations = 10  # number of pixel level iterations. The higher, the better quality
prior = 1           # for shape smoothing term. must be [0, 5]
num_levels = 50
num_histogram_bins = 50 # number of histogram bins
height, width, channels = lab.shape

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
#img_smooth_1 = cv2.resize(img_smooth,(640,480))
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
cv2.imwrite('D:\\ground_detection\\hsv_v.png',hsv[:,:,1])
cv2.destroyAllWindows()







img = cv2.imread('D:\\ground_detection\\5.png')
#img_smooth = cv2.resize(img_smooth,(360,240))
img_smooth = cv2.bilateralFilter(img, 5, 20, 20)
lab = cv2.cvtColor(img_smooth, cv2.COLOR_BGR2HSV) 
hsv = cv2.cvtColor(img_smooth, cv2.COLOR_BGR2HSV)    
num_superpixels = 200 # desired number of superpixels
num_iterations = 10  # number of pixel level iterations. The higher, the better quality
prior = 1           # for shape smoothing term. must be [0, 5]
num_levels = 50
num_histogram_bins = 50 # number of histogram bins
height, width, channels = lab.shape

seeds = cv2.ximgproc.createSuperpixelSEEDS(width, height, channels, num_superpixels, num_levels, prior, num_histogram_bins)
start_time = time.time()
seeds.iterate(lab, num_iterations)
num_of_superpixels_result = seeds.getNumberOfSuperpixels()
print('Final number of superpixels: %d' % num_of_superpixels_result)


# retrieve the segmentation result
labels = seeds.getLabels()
threshold = cp.zeros((480,640),dtype=cp.uint8)
(means, stds) = cv2.meanStdDev(lab[440:470,310:330,:])
means[2,0] = means[2,0]*0.2
features = cp.concatenate([means, stds]).flatten()
for i in range(num_of_superpixels_result):
    mask_label = 255*(labels==i)
    mask_label=mask_label.astype(cp.uint8)
    (means, stds) = cv2.meanStdDev(lab,mask=mask_label)
    means[2,0] = means[2,0]*0.2
    label_feature = cp.concatenate([means, stds]).flatten()
    d = dist.euclidean(features, label_feature)
    #if d>120:
    #    mask_label = mask_label*0
    mask_label = d*(mask_label==255)
    mask_label = mask_label.astype(cp.uint8)
    threshold = cv2.add(threshold,mask_label)
elapsed_time = time.time() - start_time
#img_smooth_1 = cv2.resize(img_smooth,(640,480))
# draw contour
mask = seeds.getLabelContourMask(False)
cv2.imshow('MaskWindow', mask)
cv2.waitKey(0)
# draw color coded image
color_img = cp.zeros((height, width, 3), cp.uint8)
color_img[:] = (0, 0, 255)
mask_inv = cv2.bitwise_not(mask)
result_bg = cv2.bitwise_and(img_smooth, img_smooth, mask=mask_inv)
result_fg = cv2.bitwise_and(color_img, color_img, mask=mask)
result = cv2.add(result_bg, result_fg)
cv2.imshow('ColorCodedWindow', result)
threshold = cv2.bitwise_not(threshold)
threshold = (threshold.astype(float)-cp.min(threshold))*255/(cp.max(threshold)-cp.min(threshold))
cv2.waitKey(0)
cv2.imshow('occupancy', threshold.astype(cp.uint8))
cv2.waitKey(0)
cv2.imwrite('D:\\ground_detection\\hsv_v.png',hsv[:,:,1])
cv2.destroyAllWindows()
