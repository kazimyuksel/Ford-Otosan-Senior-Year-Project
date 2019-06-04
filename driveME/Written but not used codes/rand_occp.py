# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 15:45:42 2019

@author: Kazım
"""

import cv2
import numpy as np

import matplotlib.pyplot as plt
import numpy as np
import random
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
from keras.preprocessing.image import ImageDataGenerator
base_img_array = np.zeros((3,492,492,3),dtype=np.uint8)

base_img_array[0,:,:,:] = cv2.resize(cv2.imread("D:/background_white.png"),(492,492))
base_img_array[1,:,:,:] = cv2.resize(cv2.imread("D:/background_hot.png"),(492,492))
base_img_array[2,:,:,:] = cv2.resize(cv2.imread("D:/background_cold.png"),(492,492))
base_occupancy = np.zeros((500,500),dtype=np.uint8)

obstacle_img_array = np.zeros((2,492,492,3),dtype=np.uint8)
obstacle_img_array[0,:,:,:] = cv2.resize(cv2.imread("D:/cardboard.png"),(492,492))
obstacle_img_array[1,:,:,:] = cv2.resize(cv2.imread("D:/cloth.png"),(492,492))

obstacle_occupancy = np.zeros((50,50,3),dtype=np.uint8)
cv2.rectangle(obstacle_occupancy, (5, 15), (45, 35), (255,255,255), -1)
cv2.imshow("asd",obstacle_occupancy)
cv2.waitKey(0)
cv2.destroyAllWindows()

obstacle_occupancy = obstacle_occupancy.reshape((1,50,50,3))
gen = ImageDataGenerator(
    rotation_range=45,
    shear_range=0.1,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=False
)
images_flow = gen.flow(obstacle_occupancy, batch_size=1)
for i, new_images in enumerate(images_flow):
    asd = new_images
    break
asd = asd[0,:,:,:]
asd = asd.astype(np.uint8)

x_offset=y_offset=50
obstacle_occupancy = obstacle_occupancy[0,:,:,:]
base_occupancy[y_offset:y_offset+asd.shape[0], x_offset:x_offset+asd.shape[1]] = asd[:,:,0]
cv2.imshow("asd",base_occupancy)
cv2.waitKey(0)
cv2.destroyAllWindows()
for i, new_images in enumerate(images_flow):
    asd = new_images
    break
asd = asd[0,:,:,:]
asd = asd.astype(np.uint8)
cv2.imshow("asd",asd)
cv2.destroyAllWindows()


def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
 
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)  
def random_float(low, high):
    return random.random()*(high-low) + low

neural_y = np.empty((0,1))
neural_x = np.empty((0,12))

gen = ImageDataGenerator(
    rotation_range=45,
    shear_range=0.2,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip = True
)
gen_2 = ImageDataGenerator(
    rotation_range=15,
    shear_range=0.2,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip = True
)
neural_x = np.zeros((200000,12),dtype=np.float64)
neural_y = np.zeros((200000,1),dtype=np.uint8)
counter=0
start_time = time.time()
for j in range(4000):
    base_occupancy = np.zeros((492,492),dtype=np.uint8)
    dst_occupancy = base_occupancy    
    base_rand = np.random.randint(0,3, size=1)[0]
    if base_rand ==0:
        dst = base_img_array[0,:,:,:]
    if base_rand ==1:
        dst = base_img_array[1,:,:,:]
    else:
        dst = base_img_array[2,:,:,:]
        
    images_flow_1 = gen_2.flow(dst.copy().reshape(1,492,492,3), batch_size=1)
    for i, new_images in enumerate(images_flow_1):
        dst_1 = new_images
        break
    dst = dst_1[0,:,:,:]
    dst = dst.astype(np.uint8)
    for i in range(25):
        base_occupancy_loop = np.zeros((492,492),dtype=np.uint8)
        obstacle_occupancy = np.zeros((50,50,3),dtype=np.uint8)
        cv2.rectangle(obstacle_occupancy, (5, 15), (45, 35), (255,255,255), -1)
        obstacle_occupancy= obstacle_occupancy.reshape((1,50,50,3))
        images_flow = gen.flow(obstacle_occupancy, batch_size=1)
        
        for i, new_images in enumerate(images_flow):
            aug_obs = new_images
            break
        aug_obs = aug_obs[0,:,:,:]
        aug_obs = aug_obs.astype(np.uint8)
        
        x_place = np.random.randint(35,465, size=1)
        y_place = np.random.randint(35,465, size=1)
        base_occupancy_loop[int(y_place[0]-aug_obs.shape[0]/2):int(y_place[0]+aug_obs.shape[0]/2),int(x_place[0]-aug_obs.shape[1]/2):int(x_place[0]+aug_obs.shape[1]/2)] = aug_obs[:,:,0]
        mask_inv = cv2.bitwise_not(base_occupancy_loop)
        # Now black-out the area of logo in ROI
        img_bg = cv2.bitwise_and(dst,dst,mask = mask_inv)
        img_bg_occupancy = cv2.bitwise_and(dst_occupancy,dst_occupancy,mask = mask_inv)
        # Take only region of logo from logo image.
    
    
        obstacle_rand = np.random.randint(0,10, size=1)[0]
        if obstacle_rand ==0:
            obs_rand = obstacle_img_array[1,:,:,:]
        else:
            obs_rand = obstacle_img_array[0,:,:,:]
        
        obs = cv2.cvtColor(obs_rand ,cv2.COLOR_BGR2HSV)
        obs_h = np.zeros((492,492),dtype=np.int16)
        obs_h_large = np.zeros((492,492),dtype=np.int16)
        hue_shift = np.random.randint(0,180, size=1)[0]
        obs_h= obs[:,:,0].astype(np.int16) + hue_shift
        mask_large = 255*(obs_h> 180)
        mask_large = mask_large.astype(np.uint8)
        mask_small = cv2.bitwise_not(mask_large)
        obs_h_large = obs_h.copy() -180
        obs_final = np.zeros((500,500),dtype=np.uint8)
        obs_final = cv2.bitwise_and(obs_h_large,obs_h_large,mask = mask_large)
        obs_final_1 = np.zeros((500,500),dtype=np.uint8)
        obs_final_1 = cv2.bitwise_and(obs_h,obs_h,mask = mask_small)
        obs_final = cv2.add(obs_final,obs_final_1)
        obs_final = obs_final.astype(np.uint8)
        obs[:,:,0] = obs_final
        gamma_side = np.random.randint(0,2, size=1)[0]
        if gamma_side ==0:
            rand_gamma = random_float(0.67, 1.0)
        else:
            rand_gamma = random_float(1.0, 1.5)
        obs[:,:,1] = adjust_gamma(obs[:,:,1], gamma=rand_gamma)
        
        gamma_side = np.random.randint(0,2, size=1)[0]
        if gamma_side ==0:
            rand_gamma = random_float(0.4, 1.0)
        else:
            rand_gamma = random_float(1.0, 2.5)
        obs[:,:,2] = adjust_gamma(obs[:,:,2], gamma=rand_gamma)

        obs = obs.astype(np.uint8)
        obs = cv2.cvtColor(obs,cv2.COLOR_HSV2BGR)
        
        img2_fg = cv2.bitwise_and(obs,obs,mask = base_occupancy_loop)
        img2_fg_occupancy =  cv2.bitwise_and(base_occupancy_loop,base_occupancy_loop,mask = base_occupancy_loop)

        dst = cv2.add(img_bg,img2_fg)
        dst_occupancy = cv2.add(img_bg_occupancy,img2_fg_occupancy)
    
    dst_bgr = cv2.GaussianBlur(dst, (5, 5), 0.5)
    gaussian = np.zeros((500,500,3),dtype=np.uint8)
    m = (0,0,0)
    gaus_noise_1 = int(np.random.randint(0,10, size=1)[0])
    gaus_noise_2 = int(np.random.randint(0,10, size=1)[0])
    gaus_noise_3 = int(np.random.randint(0,10, size=1)[0])
    s = (gaus_noise_1,gaus_noise_2,gaus_noise_3)
    cv2.randn(gaussian,m,s)
    dst[:,:,0] = cv2.add(dst_bgr[:,:,0],gaussian[:,:,0])
    dst[:,:,1] = cv2.add(dst_bgr[:,:,1],gaussian[:,:,1])
    dst[:,:,2] = cv2.add(dst_bgr[:,:,2],gaussian[:,:,2])

    dst_hsv = cv2.cvtColor(dst_bgr,cv2.COLOR_BGR2HSV)   
    
    segments = slic(dst_bgr, n_segments=200, compactness=5, sigma=0.5)
    
    print(j)
    for i in range(len(np.unique(segments))):
        ratio = 0.5
        mask_label = 255*(segments==i)
        mask_label=mask_label.astype(np.uint8)
        (means_bgr, stds_bgr) = cv2.meanStdDev(dst_bgr,mask=mask_label)
        label_feature = np.concatenate([means_bgr, stds_bgr])
        label_feature = label_feature / 255.0
        (means_hsv, stds_hsv) = cv2.meanStdDev(dst_hsv,mask=mask_label)
        means_hsv[0,0] = means_hsv[0,0] / 180.0
        means_hsv[1,0] = means_hsv[1,0] / 255.0
        means_hsv[2,0] = means_hsv[2,0] / 255.0
        stds_hsv = stds_hsv / 255.0
        label_feature = np.concatenate([label_feature, means_hsv, stds_hsv]).flatten()
        label_feature = label_feature.reshape((1,12))
        sum_mask = np.sum(mask_label)
        occupancy =  cv2.bitwise_and(dst_occupancy,dst_occupancy,mask = mask_label)
        sum_occupancy = np.sum(occupancy)
        ratio = sum_occupancy/sum_mask
        add2array = np.random.randint(0,10, size=1)[0]
        add2array = add2array / 10.0
        if ratio < 0.08 and add2array<0.1:
            neural_x[counter,:] = label_feature
            neural_y[counter,:] = 0
            counter = counter +1
        elif ratio > 0.92:
            neural_x[counter,:] = label_feature
            neural_y[counter,:] = 1
            counter = counter +1
elapsed_time = time.time() - start_time
import h5py
with h5py.File('occp2.h5','w') as hf:
    hf.create_dataset('X', data=neural_x)
    hf.create_dataset('Y', data=neural_y)
