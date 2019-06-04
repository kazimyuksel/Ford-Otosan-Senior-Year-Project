# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 16:50:08 2019

@author: Kazım
"""

import os
import sys
import math
import datetime
import time

import cv2
import numpy as np
from colorama import Fore, Style
from skimage.measure import block_reduce as pooling
from skimage import color,io,exposure,transform
from sklearn.cluster import KMeans
from progressbar import ProgressBar, Bar, AdaptiveETA, Percentage, SimpleProgress, Timer
from scipy.spatial import distance as dist

video_capture = cv2.VideoCapture(1)
video_capture.set(3,640)
video_capture.set(4,480)
video_capture.set(5,20)
   
# set parameters for superpixel segmentation
num_superpixels = 500  # desired number of superpixels
num_iterations = 50     # number of pixel level iterations. The higher, the better quality
prior = 5              # for shape smoothing term. must be [0, 5]
num_levels = 15
num_histogram_bins = 15 # number of histogram bins
height, width, channels = (480,640,3)

# initialize SEEDS algorithm
seeds = cv2.ximgproc.createSuperpixelSEEDS(width, height, channels, num_superpixels, num_levels, prior, num_histogram_bins)


start_time = time.time()
avg_hz = 0
while True:
    ret, rgb = video_capture.read()
    img_smooth = cv2.bilateralFilter(rgb, 5, 20, 20)
    lab = cv2.cvtColor(img_smooth, cv2.COLOR_BGR2HSV) 	
    seeds.iterate(lab,num_iterations)
    num_of_superpixels_result = seeds.getNumberOfSuperpixels()
    #print('Final number of superpixels: %d' % num_of_superpixels_result)
    

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

    threshold = cv2.bitwise_not(threshold)
    threshold = (threshold.astype(float)-np.min(threshold))*255/(np.max(threshold)-np.min(threshold))
    cv2.imshow('occupancy', threshold.astype(np.uint8))
    elapsed_time = time.time() - start_time
    start_time =time.time()
    avg_hz = 0.8*avg_hz + 0.2/elapsed_time
    print(avg_hz)
    if cv2.waitKey(27) & 0xFF == ord('q') :
        break
    
video_capture.release()
cv2.destroyAllWindows()