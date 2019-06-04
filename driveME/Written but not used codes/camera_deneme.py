# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 17:45:46 2019

@author: Kazım
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 16:50:08 2019

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

video_capture = cv2.VideoCapture(0)
video_capture.set(3,640)
video_capture.set(4,480)
video_capture.set(5,20)

width = 640
height = 480
scaling_ratio = 1
new_width = scaling_ratio*width
new_width = (int)(new_width)
new_height = scaling_ratio*height
new_height = (int)(new_height)

start_time = time.time()
avg_hz = 0
while True:
    ret, rgb = video_capture.read()
    img_smooth = cv2.bilateralFilter(rgb, 5, 20, 20)
    gray = cv2.cvtColor(img_smooth, cv2.COLOR_BGR2GRAY) 
    img_orj = img_smooth
    img_smooth = cv2.resize(img_smooth,(new_width,new_height))    
    img_smooth = cv2.cvtColor(img_smooth, cv2.COLOR_BGR2RGB) 
    img_smooth = img_as_float(img_smooth)
    
    segments_fz = felzenszwalb(img_smooth, scale=100, sigma=0.5, min_size=50)
    threshold = np.zeros((new_height,new_width),dtype=np.uint8)
    (means, stds) = cv2.meanStdDev(img_orj[440:470,310:330,:])
    means[2,0] = means[2,0]
    means[1,0] = means[1,0]
    features = np.concatenate([means, stds]).flatten()
    for i in range(len(np.unique(segments_fz))):
        mask_label = 255*(segments_fz==i)
        mask_label=mask_label.astype(np.uint8)
        (means, stds) = cv2.meanStdDev(img_smooth,mask=mask_label)
        means[2,0] = means[2,0]
        means[1,0] = means[1,0]
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
    #out = cv2.bilateralFilter(out, 10, 75, 75)
    out = cv2.resize(out,(640,480))
    cv2.imshow('occupancy', out)
    ret,thresh1 = cv2.threshold(out,220,255,cv2.THRESH_BINARY)
    kernel = np.ones((3,3),np.uint8)
    thresh1 = cv2.dilate(thresh1,kernel,iterations = 2)
    thresh1 = cv2.erode(thresh1,kernel,iterations = 2)
    
    cv2.imshow('threshold', thresh1)
    elapsed_time = time.time() - start_time
    start_time =time.time()
    avg_hz = 0.8*avg_hz + 0.2/elapsed_time
    print(avg_hz)
    if cv2.waitKey(27) & 0xFF == ord('q') :
        break
    
video_capture.release()
cv2.destroyAllWindows()