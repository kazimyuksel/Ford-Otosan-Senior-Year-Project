# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 12:21:49 2018

@author: Kazım
"""
import cv2
import numpy as np
import pandas as pd
camera_height = 75 #cm
camera_theta =26.15  

camera_H_angle = 99.64 #degree
camera_W_angle = 100 #degree
compression_factor_H = 1
compression_factor_W = 1
img = cv2.imread('D:\\undistorted.jpg',-1)
#img = cv2.resize(img, (960, 540)) 

H,W,_ = np.shape(img)
L_ref = 2*camera_height*np.tan(np.radians(camera_H_angle)/5)
h = H/np.radians(camera_H_angle)
w = W/np.radians(camera_W_angle)
x_mid = camera_height/np.tan(np.radians(camera_theta))
#phi_1 = np.arctan(camera_height/(x_mid-L_ref/2)) - np.radians(camera_theta)
#phi_2 = np.radians(camera_theta) - np.arctan(camera_height/(x_mid+L_ref/2))

phi_1 = np.arccos((camera_height*camera_height + x_mid*(x_mid-L_ref/2))/((np.sqrt(np.square(camera_height)+np.square(x_mid)))*(np.sqrt(np.square(camera_height)+np.square(x_mid-L_ref/2)))))
phi_2 = np.arccos((camera_height*camera_height + x_mid*(x_mid+L_ref/2))/((np.sqrt(np.square(camera_height)+np.square(x_mid)))*(np.sqrt(np.square(camera_height)+np.square(x_mid+L_ref/2)))))

beta_1 = np.arctan((L_ref/2)/(np.sqrt(np.square(x_mid-L_ref/2)+np.square(camera_height))))
beta_2 = np.arctan((L_ref/2)/(np.sqrt(np.square(x_mid+L_ref/2)+np.square(camera_height))))

#beta_1 = np.arctan((L_ref/2)/(x_mid-L_ref/2))
#beta_2 = np.arctan((L_ref/2)/(x_mid+L_ref/2))
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
pts2 = np.float32([[540,460],[540,540],[460,540],[460,460]])

matrix = cv2.getPerspectiveTransform(pts1,pts2)
result = cv2.warpPerspective(img, matrix, (1000,1000))
#cv2.circle(result, (495,780),5, (0,255,0), thickness=15)

M = cv2.getRotationMatrix2D((1000/2,1000/2),-45,1)
dst = cv2.warpAffine(result,M,(1000,1000))
thres = 50



dst = cv2.cvtColor(dst, cv2.COLOR_BGR2HSV)
sat = dst[:,:,1]
val = dst[:,:,2]
ret,mask = cv2.threshold(val,10,127,cv2.THRESH_BINARY_INV)
ret,bin_inv_dst = cv2.threshold(sat,thres,255,cv2.THRESH_BINARY_INV)
bin_inv_dst = bin_inv_dst - mask


img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
sat = img[:,:,1]
val = img[:,:,2]
ret,mask = cv2.threshold(val,10,127,cv2.THRESH_BINARY_INV)
ret,bin_inv_img = cv2.threshold(sat,thres,255,cv2.THRESH_BINARY_INV)
bin_inv_img = bin_inv_img - mask

result = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
sat = result[:,:,1]
val = result[:,:,2]
ret,mask = cv2.threshold(val,10,127,cv2.THRESH_BINARY_INV)
ret,bin_inv_result = cv2.threshold(sat,thres,255,cv2.THRESH_BINARY_INV)
bin_inv_result = bin_inv_result - mask

#cv2.imshow('mask',mask)
cv2.imshow('image',bin_inv_result)
cv2.imshow('original',bin_inv_img)
cv2.imshow('original',result)
#cv2.imshow('original',img)
#cv2.imshow('rotated',bin_inv_dst)

#cv2.imwrite('D:\\corner_birdview_rotated_saturation_bin_inv.jpg',bin_inv_dst)

#cv2.imwrite('D:\\corner_withline_saturation_bin_inv.jpg',bin_inv_img)

#cv2.imwrite('D:\\corner_birdview_saturation_bin_inv.jpg',bin_inv_result)

cv2.waitKey(0)
cv2.destroyAllWindows()


#a = (np.sqrt(np.square(camera_height)+np.square(x_mid)))*(np.sqrt(np.square(camera_height)+np.square(x_mid-L_ref/2)))
#b = (np.sqrt(np.square(camera_height)+np.square(x_mid)))*(np.sqrt(np.square(camera_height)+np.square(x_mid+L_ref/2)))