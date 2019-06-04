# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 14:31:19 2019

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
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float

video_capture = cv2.VideoCapture(1)
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

DIM = (640,480)
K=np.array([[433.1789201825628, 0.0, 336.0326551869997], [0.0, 430.52735997037956, 216.5706380556904], [0.0, 0.0, 1.0]])
D=np.array([[-0.05280848062554133], [-0.13174782579340452], [-0.028567480262252885], [0.060944836127339065]])

dim1 = DIM
dim2 = (640, 480)
dim3 = (640, 480)
balance=1.0

bool_start_1 = True
bool_start_2 = True

_width  = 35.0
_height = 35.0
_margin = 235.0

pts_dst_corner = np.array(
	[
		[[  		_margin, _margin 			]],
		[[ 			_margin, _height + _margin  ]],
		[[ _width + _margin, _height + _margin  ]],
		[[ _width + _margin, _margin 			]],
	]
)
pts_dst = np.array( pts_dst_corner, np.float32 )
scaled_K = K * dim1[0] / DIM[0]  # The values of K is to scale with image dimension.
scaled_K[2][2] = 1.0  # Except that K[2][2] is always 1.0
# This is how scaled_K, dim2 and balance are used to determine the final K used to un-distort image. OpenCV document failed to make this clear!
new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, D, dim2, np.eye(3), balance=balance)
map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, D, np.eye(3), new_K, dim3, cv2.CV_16SC2)

aruco_dict = aruco.Dictionary_get( aruco.DICT_4X4_1000 )
markerLength = 35.0
arucoParams = aruco.DetectorParameters_create()



criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-12)
counter = 1


corners_array = np.empty((0,4,2))

start_time = time.time()
avg_hz = 0
while True:
    ret, rgb = video_capture.read()
    rgb = cv2.remap(rgb, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    
    img_smooth = cv2.bilateralFilter(rgb, 5, 20, 20)
    gray = cv2.cvtColor(img_smooth, cv2.COLOR_BGR2GRAY) 
    img_orj = img_smooth
    img_smooth = cv2.resize(img_smooth,(new_width,new_height)) 
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=arucoParams)
    if ids is not None and counter is not 100:
        imgWithAruco = aruco.drawDetectedMarkers(rgb, corners, ids, (0,255,0))
        corners_array = np.concatenate([corners_array,corners[0]],axis = 0)
        counter = counter +1
    else:
        imgWithAruco = rgb
    out = cv2.resize(gray,(640,480))
    cv2.imshow('occupancy', imgWithAruco)
    elapsed_time = time.time() - start_time
    start_time =time.time()
    avg_hz = 0.8*avg_hz + 0.2/elapsed_time
    print(avg_hz)
    if counter is 100:
        mean_corner = corners_array.mean(axis=0)
        break
    if cv2.waitKey(27) & 0xFF == ord('q') :
        break
    
video_capture.release()
cv2.destroyAllWindows()













video_capture = cv2.VideoCapture(1)
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

DIM = (640,480)
K=np.array([[433.1789201825628, 0.0, 336.0326551869997], [0.0, 430.52735997037956, 216.5706380556904], [0.0, 0.0, 1.0]])
D=np.array([[-0.05280848062554133], [-0.13174782579340452], [-0.028567480262252885], [0.060944836127339065]])

dim1 = DIM
dim2 = (640, 480)
dim3 = (640, 480)
balance=1.0

bool_start_1 = True
bool_start_2 = True

_width  = 35.0
_height = 35.0
_margin = 465.0

pts_dst_corner = np.array(
	[
		[[  		_margin, _margin 			]],
		[[ 			_margin, _height + _margin  ]],
		[[ _width + _margin, _height + _margin  ]],
		[[ _width + _margin, _margin 			]],
	]
)
pts_dst = np.array( pts_dst_corner, np.float32 )
scaled_K = K * dim1[0] / DIM[0]  # The values of K is to scale with image dimension.
scaled_K[2][2] = 1.0  # Except that K[2][2] is always 1.0
# This is how scaled_K, dim2 and balance are used to determine the final K used to un-distort image. OpenCV document failed to make this clear!
new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, D, dim2, np.eye(3), balance=balance)
map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, D, np.eye(3), new_K, dim3, cv2.CV_16SC2)

aruco_dict = aruco.Dictionary_get( aruco.DICT_4X4_1000 )
markerLength = 35.0
arucoParams = aruco.DetectorParameters_create()



criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-12)
counter = 1


corners_array = np.empty((0,4,2))

start_time = time.time()
avg_hz = 0
pts_src = mean_corner.reshape((4,1,2))
pts_src = np.rollaxis(pts_src,1)
h, status = cv2.findHomography( pts_src, pts_dst )
print_lpf = np.zeros((500,500),dtype=np.uint8)
while True:
    ret, rgb = video_capture.read()

    rgb = cv2.remap(rgb, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    
    img_smooth = cv2.bilateralFilter(rgb, 5, 20, 20)
    gray = cv2.cvtColor(img_smooth, cv2.COLOR_BGR2GRAY) 
    img_orj = img_smooth
    out = cv2.warpPerspective( img_orj, h, ( int( _width + _margin * 2 ), int( _height + _margin * 2 ) ) )
    out = out[240:600,240:600,:]
    gray_warp = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    img_smooth = cv2.cvtColor(out, cv2.COLOR_BGR2RGB) 
    img_smooth = cv2.resize(img_smooth,(200,200))
    img_compare = img_smooth.copy()
    img_smooth = img_as_float(img_smooth)
    
    
    #segments_fz = slic(img_smooth, n_segments=200, compactness=10, sigma=0.5)    
    threshold = np.zeros((200,200),dtype=np.uint8)
    #segments_fz = slic(img_smooth, n_segments=200, compactness=10, sigma=0.4)
    segments_fz = felzenszwalb(img_smooth, scale=1000, sigma=0.8, min_size=100)
    #segments_fz = quickshift(img_smooth, kernel_size=3, max_dist=6, ratio=0.5)

    
    (means, stds) = cv2.meanStdDev(img_compare[55:65,55:65,:])
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
    
    
    #threshold = cv2.bitwise_not(threshold)
    threshold = (threshold.astype(float)-np.min(threshold))*255/(np.max(threshold)-np.min(threshold))
    out_sp=cv2.convertScaleAbs(threshold.astype(np.uint8))
    out_sp = cv2.bilateralFilter(out_sp, 10, 75, 75)
    out_sp = cv2.resize(out_sp,(500,500))
    #cv2.imshow('occupancy', out)
    ret,thresh1 = cv2.threshold(out_sp,220,255,cv2.THRESH_BINARY)
    kernel = np.ones((3,3),np.uint8)
    thresh1 = cv2.dilate(thresh1,kernel,iterations = 2)
    thresh1 = cv2.erode(thresh1,kernel,iterations = 2)
    threshold = threshold.astype(np.uint8)
    threshold = cv2.resize(threshold,(500,500))
    print_lpf = 0.4*print_lpf + 0.6*threshold.astype(np.uint8)
    print_lpf = (print_lpf-np.min(print_lpf))*255/(np.max(print_lpf)-np.min(print_lpf))
    cv2.imshow('threshold', print_lpf.astype(np.uint8))
    out_sclup = mark_boundaries(img_smooth, segments_fz)
    out_sclup = out_sclup*255
    out_sclup = out_sclup.astype(np.uint8)
    out_sclup = cv2.resize(out,(500,500))
    cv2.imshow('occupancy', out_sclup)
    elapsed_time = time.time() - start_time
    start_time =time.time()
    avg_hz = 0.8*avg_hz + 0.2/elapsed_time
    print(avg_hz)
    if cv2.waitKey(27) & 0xFF == ord('q') :
        break
    
video_capture.release()
cv2.destroyAllWindows()













video_capture = cv2.VideoCapture(1)
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

DIM = (640,480)
K=np.array([[433.1789201825628, 0.0, 336.0326551869997], [0.0, 430.52735997037956, 216.5706380556904], [0.0, 0.0, 1.0]])
D=np.array([[-0.05280848062554133], [-0.13174782579340452], [-0.028567480262252885], [0.060944836127339065]])

dim1 = DIM
dim2 = (640, 480)
dim3 = (640, 480)
balance=1.0

bool_start_1 = True
bool_start_2 = True

_width  = 15.0
_height = 15.0
_margin = 235.0

pts_dst_corner = np.array(
	[
		[[  		_margin, _margin 			]],
		[[ 			_margin, _height + _margin  ]],
		[[ _width + _margin, _height + _margin  ]],
		[[ _width + _margin, _margin 			]],
	]
)
pts_dst = np.array( pts_dst_corner, np.float32 )
scaled_K = K * dim1[0] / DIM[0]  # The values of K is to scale with image dimension.
scaled_K[2][2] = 1.0  # Except that K[2][2] is always 1.0
# This is how scaled_K, dim2 and balance are used to determine the final K used to un-distort image. OpenCV document failed to make this clear!
new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, D, dim2, np.eye(3), balance=balance)
map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, D, np.eye(3), new_K, dim3, cv2.CV_16SC2)

aruco_dict = aruco.Dictionary_get( aruco.DICT_6X6_1000 )
markerLength = 15.0
arucoParams = aruco.DetectorParameters_create()



criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-12)
counter = 1
start_time = time.time()
avg_hz = 0
while True:
    ret, rgb = video_capture.read()
    rgb = cv2.remap(rgb, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
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