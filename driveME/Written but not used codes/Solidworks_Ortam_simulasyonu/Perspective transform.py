# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 15:50:04 2018

@author: Kazım
"""
import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

camera_height = 150 #cm
camera_theta =26.15 

camera_H_angle = 89 #degree
camera_W_angle = 120 #degree
compression_factor_H = 1
compression_factor_W = 1
img = cv2.imread('D:\\normal.jpg',-1)
#img = cv2.resize(img, (960, 540)) 

h,w = img.shape[:2]
im_small_long = img.reshape((h * w, 3))
im_small_wide = im_small_long.reshape((h,w,3))

km = KMeans(n_clusters=3)

km.fit(im_small_long)

cc = km.cluster_centers_.astype(np.uint8)
img = np.asarray([cc[i] for i in km.labels_]).reshape((h,w,3))


H,W,_ = np.shape(img)
L_ref = 2*camera_height*np.tan(np.radians(camera_H_angle)/4)
h = H/np.radians(camera_H_angle)
w = W/np.radians(camera_W_angle)
x_mid = camera_height/np.tan(np.radians(camera_theta))
#phi_1 = np.arctan(camera_height/(x_mid-L_ref/2)) - np.radians(camera_theta)
#phi_2 = np.radians(camera_theta) - np.arctan(camera_height/(x_mid+L_ref/2))

phi_1 = np.arccos((camera_height*camera_height + x_mid*(x_mid-L_ref/2))/((np.sqrt(np.square(camera_height)+np.square(x_mid)))*(np.sqrt(np.square(camera_height)+np.square(x_mid-L_ref/2)))))
phi_2 = np.arccos((camera_height*camera_height + x_mid*(x_mid+L_ref/2))/((np.sqrt(np.square(camera_height)+np.square(x_mid)))*(np.sqrt(np.square(camera_height)+np.square(x_mid+L_ref/2)))))


beta_1 = np.arctan((L_ref/2)/(np.sqrt(np.square(x_mid-L_ref/2)+np.square(camera_height))))
beta_2 = np.arctan((L_ref/2)/(np.sqrt(np.square(x_mid+L_ref/2)+np.square(camera_height))))
I_t = H/2 - compression_factor_H*phi_1*h
I_b = H/2 + compression_factor_H*phi_2*h
J_tr = W/2 + compression_factor_W*beta_2*w
J_tl = W/2 - compression_factor_W*beta_2*w
J_br = W/2 + compression_factor_W*beta_1*w
J_bl = W/2 - compression_factor_W*beta_1*w

p_1_prime = (np.round(J_tr).astype(int),np.round(I_t).astype(int))
p_2_prime = (np.round(J_br).astype(int),np.round(I_b).astype(int))
p_3_prime = (np.round(J_bl).astype(int),np.round(I_b).astype(int))
p_4_prime = (np.round(J_tl).astype(int),np.round(I_t).astype(int))


cv2.line(img, p_1_prime, p_2_prime, (0,255,0), thickness=2, lineType=8, shift=0)
cv2.line(img, p_2_prime, p_3_prime, (0,255,0), thickness=2, lineType=8, shift=0)
cv2.line(img, p_3_prime, p_4_prime, (0,255,0), thickness=2, lineType=8, shift=0)
cv2.line(img, p_4_prime, p_1_prime, (0,255,0), thickness=2, lineType=8, shift=0)

#cv2.circle(img, p_3_prime,5, (0,255,0), thickness=15)
#cv2.circle(img, p_4_prime,5, (0,255,0), thickness=15)
#cv2.circle(img, p_2_prime,5, (0,255,0), thickness=1, lineType=8, shift=0)
#cv2.circle(img, p_3_prime,5, (0,255,0), thickness=1, lineType=8, shift=0)
#cv2.circle(img, p_4_prime,5, (0,255,0), thickness=1, lineType=8, shift=0)

pts1 = np.float32([[J_tr,I_t],[J_br,I_b],[J_bl,I_b],[J_tl,I_t]])
pts2 = np.float32([[520,484],[520,516],[480,516],[480,484]])

matrix = cv2.getPerspectiveTransform(pts1,pts2)
result = cv2.warpPerspective(img, matrix, (1000,1000))

cv2.imshow('image',result)
cv2.imshow('original',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#a = (np.sqrt(np.square(camera_height)+np.square(x_mid)))*(np.sqrt(np.square(camera_height)+np.square(x_mid-L_ref/2)))
#b = (np.sqrt(np.square(camera_height)+np.square(x_mid)))*(np.sqrt(np.square(camera_height)+np.square(x_mid+L_ref/2)))