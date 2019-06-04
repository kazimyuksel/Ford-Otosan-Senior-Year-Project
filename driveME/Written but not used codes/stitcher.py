# -*- coding: utf-8 -*-
"""
Created on Tue May 21 23:37:59 2019

@author: Kazım
"""
import cv2
import numpy as np
import time
img_1 = cv2.imread("cam_1.png",1)
img_2 = cv2.imread("cam_2.png",1)
img_3 = cv2.imread("cam_3.png",1)
img_4 = cv2.imread("cam_4.png",1)
start_time = time.time()
port_shape = (492,492)
center = int(port_shape[1]/2)
mask_width = 40.0
temp_img_1 = np.zeros((492,492,3))
temp_img_2 = np.zeros((492,492,3))
img_final = np.zeros((492,492,3))

img_mask = np.ones((492,492,3),dtype=np.float64)
img_mask[:int(center+mask_width/2),:,:] = 0.0
mask_step = 1.0/mask_width
for j in range(int(mask_width)):
    start_pos = int(center-mask_width/2)
    img_mask[int(j+start_pos),:,:] = j*mask_step
    
img_1_masked = np.multiply(img_1,img_mask)
img_2_masked = np.multiply(img_2,img_mask)
img_3_masked = np.multiply(img_3,img_mask)
img_4_masked = np.multiply(img_4,img_mask)

img_2_masked = np.rot90(img_2_masked,k=1)
img_3_masked = np.rot90(img_3_masked,k=2)
img_4_masked = np.rot90(img_4_masked,k=3)

temp_img_1 = cv2.add(img_1_masked,img_3_masked)
temp_img_2 = cv2.add(img_2_masked,img_4_masked)

img_final = cv2.addWeighted(temp_img_1,0.5,temp_img_2,0.5,0)

out = cv2.subtract(img_final, temp_img_1)
out = 255*(out-out.min())/(out.max()-out.min())
print(time.time()-start_time)
cv2.imwrite("masked_cam.png",out.astype(np.uint8))