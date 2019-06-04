# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 20:18:08 2019

@author: Kazım
"""

import glob
import numpy as np
import cv2
import os 
import time
from io import BytesIO
import redis
from scipy import ndimage

def imgShow(img):
    try:
        cv2.imshow("image",img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
dim = 492
store = redis.Redis(host="192.168.1.101",port=6379)
prev_image_id_1 = None
prev_image_id_2 = None
prev_image_id_3 = None
prev_image_id_4 = None
prev_data_id = None

def getData(prev_image_id_1,prev_image_id_2,prev_image_id_3,prev_image_id_4, prev_data_id):
    MAX_FPS = 50
    image_id = None
    while True:
        time.sleep(1./MAX_FPS)
        image_id = store.get('image_id_1')
        if image_id != prev_image_id_1:
            break
    prev_image_id_1 = image_id
    image = store.get('image_1')
    image = BytesIO(image)
    image = np.load(image)
    img_1 = cv2.imdecode(image, -1)
    image_id = None
    while True:
        time.sleep(1./MAX_FPS)
        image_id = store.get('image_id_2')
        if image_id != prev_image_id_2:
            break
    prev_image_id_2 = image_id
    image = store.get('image_2')
    image = BytesIO(image)
    image = np.load(image)
    img_2 = cv2.imdecode(image, -1)
    image_id = None
    while True:
        time.sleep(1./MAX_FPS)
        image_id = store.get('image_id_3')
        if image_id != prev_image_id_3:
            break
    prev_image_id_3 = image_id
    image = store.get('image_3')
    image = BytesIO(image)
    image = np.load(image)
    img_3 = cv2.imdecode(image, -1)
    image_id = None
    while True:
        time.sleep(1./MAX_FPS)
        image_id = store.get('image_id_4')
        if image_id != prev_image_id_4:
            break
    prev_image_id_4 = image_id
    image = store.get('image_4')
    image = BytesIO(image)
    image = np.load(image)
    img_4 = cv2.imdecode(image, -1)
    
    position_data_id = None
    while True:
        time.sleep(0.05)
        position_data_id = store.get('master_spatial_attributes_id')
        if position_data_id != prev_data_id:
            break
    prev_data_id = position_data_id
    byte_get = store.get('master_spatial_attributes_data')
    byte_get = BytesIO(byte_get)
    float_get = np.load(byte_get)

    headPos = (float_get[0],float_get[1])
    headAngle = float_get[2]
    trailerPos = (float_get[3],float_get[4])
    trailerAngle = float_get[5]
    
    return prev_image_id_1, prev_image_id_2, prev_image_id_3, prev_image_id_4, prev_data_id, img_1, img_2, img_3, img_4, headPos, trailerPos, headAngle, trailerAngle


def carRemove(image,headPos,trailerPos,headAngle,trailerAngle):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    headPos = (headPos[0]*(image.shape[0]/492.0),headPos[1]*(image.shape[0]/492.0))
    trailerPos = (trailerPos[0]*(image.shape[0]/492.0),trailerPos[1]*(image.shape[0]/492.0))
    headPos = (headPos[0],image.shape[0]-headPos[1])
    trailerPos = (trailerPos[0],image.shape[0]-trailerPos[1])
    mask = np.zeros((image.shape),dtype = np.float64)
    trailer_length = 95*(image.shape[0]/492.0)
    trailer_width =  20*(image.shape[0]/492.0)
    trailer_w2front = 66.45*(image.shape[0]/492.0)
    trailer_w2back = 28.55*(image.shape[0]/492.0)
    trailer_min_pad_size = np.sqrt((trailer_width/2)**2+trailer_w2front**2)
    trailer = np.ones((int(trailer_width),int(trailer_length)))
    trailer = np.pad(trailer,((int(trailer_min_pad_size-trailer_width/2),int(trailer_min_pad_size-trailer_width/2)),(int(trailer_min_pad_size-trailer_w2back),int(trailer_min_pad_size-trailer_w2front))),"constant",constant_values=(0,0))
    trailer = ndimage.rotate(trailer,trailerAngle*180.0/np.pi,reshape=False)
    trailer = 255*(trailer-trailer.min())/(trailer.max()-trailer.min())
    _,trailer = cv2.threshold(trailer,120,255,cv2.THRESH_BINARY)
    trailer = trailer.astype(np.uint8)
    
    
    head_length = 43*(image.shape[0]/492.0)
    head_width = 20*(image.shape[0]/492.0)
    head_a2front = 38*(image.shape[0]/492.0)
    head_a2back = 5*(image.shape[0]/492.0)
    head_min_pad_size = np.sqrt((head_width/2)**2+head_a2front**2)
    head = np.ones((int(head_width),int(head_length)))
    head = np.pad(head,((int(head_min_pad_size-head_width/2),int(head_min_pad_size-head_width/2)),(int(head_min_pad_size-head_a2back),int(head_min_pad_size-head_a2front))),"constant",constant_values=(0,0))
    head = ndimage.rotate(head,headAngle*180.0/np.pi,reshape=False)
    head = 255*(head-head.min())/(head.max()-head.min())
    _,head = cv2.threshold(head,120,255,cv2.THRESH_BINARY)
    head = head.astype(np.uint8)
    
    
    max_pad = np.ceil(max(head_min_pad_size,trailer_min_pad_size))
    roi_head_topleft = (int(max_pad+headPos[0]-head.shape[1]/2),int(max_pad+headPos[1]-head.shape[0]/2))
    roi_trailer_topleft = (int(max_pad+trailerPos[0]-trailer.shape[1]/2),int(max_pad+trailerPos[1]-trailer.shape[0]/2))
    mask = np.pad(mask,((int(max_pad),int(max_pad)),(int(max_pad),int(max_pad))),"constant",constant_values=(0,0))
    mask_h=mask.copy()
    mask_t=mask.copy()
    
    mask_h[roi_head_topleft[1]:roi_head_topleft[1]+head.shape[0],roi_head_topleft[0]:roi_head_topleft[0]+head.shape[1]] =+ head
    mask_t[roi_trailer_topleft[1]:roi_trailer_topleft[1]+trailer.shape[0],roi_trailer_topleft[0]:roi_trailer_topleft[0]+trailer.shape[1]] =+ trailer
    mask = cv2.bitwise_or(mask_t,mask_h)
    mask = mask[int(max_pad):int(max_pad+image.shape[0]),int(max_pad):int(max_pad+image.shape[1])]
    mask = cv2.dilate(mask,kernel,iterations=2).astype(np.uint8)
    mask = cv2.bitwise_not(mask)
    mask = mask/255.0
    mask = np.multiply(image,mask)
    mask = mask.astype(np.uint8)
    return mask

def occupancyMap(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    image = cv2.bilateralFilter(cv2.bilateralFilter(image,9,75,25),9,75,75)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    blur = cv2.GaussianBlur(cv2.bilateralFilter(image[:,:,1],9,50,50),(5,5),1)
    ret3,blur = cv2.threshold(blur, 80, 80, cv2.THRESH_TRUNC )
    ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    th3 = cv2.morphologyEx(th3, cv2.MORPH_OPEN, kernel)
    return th3

def createIntersectionMask(img_1,img_2,img_3,img_4):
    img_2 = ndimage.rotate(img_2,90)
    img_3 = ndimage.rotate(img_3,180)
    img_4 = ndimage.rotate(img_4,270)
    _,img_1_mask = cv2.threshold(cv2.cvtColor(img_1,cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY)
    _,img_2_mask = cv2.threshold(cv2.cvtColor(img_2,cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY)
    _,img_3_mask = cv2.threshold(cv2.cvtColor(img_3,cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY)
    _,img_4_mask = cv2.threshold(cv2.cvtColor(img_4,cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY)
    #mask 1,2,3,4
    center13 = cv2.bitwise_and(img_1_mask,img_3_mask)
    center24 = cv2.bitwise_and(img_2_mask,img_4_mask)
    mask1234 = cv2.bitwise_and(center13,center24)
    #mask 1,2
    mask12 = cv2.bitwise_and(cv2.bitwise_not(img_4_mask),cv2.bitwise_not(img_3_mask))
    #mask 1,2
    mask23 = cv2.bitwise_and(cv2.bitwise_not(img_1_mask),cv2.bitwise_not(img_4_mask))
    #mask 1,2
    mask34 = cv2.bitwise_and(cv2.bitwise_not(img_2_mask),cv2.bitwise_not(img_1_mask))
    #mask 1,2
    mask14 = cv2.bitwise_and(cv2.bitwise_not(img_2_mask),cv2.bitwise_not(img_3_mask))
    #mask 1,3,4
    mask134 = cv2.bitwise_and(cv2.bitwise_not(img_2_mask),center13)
    #mask 1,2,4
    mask124 = cv2.bitwise_and(cv2.bitwise_not(img_3_mask),center24)
    #mask 1,2,3
    mask123 = cv2.bitwise_and(cv2.bitwise_not(img_4_mask),center13)
    #mask 2,3,4
    mask234 = cv2.bitwise_and(cv2.bitwise_not(img_1_mask),center24)
    return mask124, mask12, mask123, mask23, mask234, mask34, mask134, mask14, mask1234

prev_image_id_1, prev_image_id_2, prev_image_id_3, prev_image_id_4, prev_data_id, img_1, img_2, img_3, img_4, headPos, trailerPos, headAngle, trailerAngle = getData(prev_image_id_1,prev_image_id_2,prev_image_id_3,prev_image_id_4, prev_data_id)
mask124, mask12, mask123, mask23, mask234, mask34, mask134, mask14, mask1234 = createIntersectionMask(img_1,img_2,img_3,img_4)
while True:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    occp_final = np.zeros((dim,dim),dtype=np.uint8)
    prev_image_id_1, prev_image_id_2, prev_image_id_3, prev_image_id_4, prev_data_id, img_1, img_2, img_3, img_4, headPos, trailerPos, headAngle, trailerAngle = getData(prev_image_id_1,prev_image_id_2,prev_image_id_3,prev_image_id_4, prev_data_id)
    occp_1 = occupancyMap(img_1)
    occp_2 = ndimage.rotate(occupancyMap(img_2),90)
    occp_3 = ndimage.rotate(occupancyMap(img_3),180)
    occp_4 = ndimage.rotate(occupancyMap(img_4),270)
    
    occp_1234 = cv2.bitwise_and(cv2.bitwise_and(occp_1,occp_3),cv2.bitwise_and(occp_2,occp_4),mask = mask1234)
    
    occp_12 = cv2.bitwise_and(occp_1,occp_2,mask = mask12)
    occp_23 = cv2.bitwise_and(occp_2,occp_3,mask = mask23)
    occp_34 = cv2.bitwise_and(occp_3,occp_4,mask = mask34)
    occp_14 = cv2.bitwise_and(occp_1,occp_4,mask = mask14)
    
    occp_123 = cv2.bitwise_and(cv2.bitwise_and(occp_1,occp_3),occp_2,mask = mask123)
    occp_234 = cv2.bitwise_and(cv2.bitwise_and(occp_2,occp_3),occp_2,mask = mask234)
    occp_124 = cv2.bitwise_and(cv2.bitwise_and(occp_1,occp_2),occp_4,mask = mask124)
    occp_134 = cv2.bitwise_and(cv2.bitwise_and(occp_1,occp_3),occp_4,mask = mask134)
    occp_final = occp_12+occp_23+occp_34+occp_14+occp_123+occp_234+occp_124+occp_134+occp_1234
    occp_final = occp_final.astype(np.float64)
    occp_final = carRemove(occp_final,headPos,trailerPos,headAngle,trailerAngle)

    
    port_shape = (492,492)
    center = int(port_shape[1]/2)
    mask_width = 40.0
    img_mask = np.ones((492,492,3),dtype=np.float64)
    img_mask[:int(center+mask_width/2),:,:] = 0.0
    mask_step = 1.0/mask_width
    for j in range(int(mask_width)):
        start_pos = int(center-mask_width/2)
        img_mask[int(j+start_pos),:,:] = j*mask_step
    img_mask = cv2.resize(img_mask,(dim,dim))
    img_1 = img_1.astype(np.float64)
    img_2 = img_2.astype(np.float64)
    img_3 = img_3.astype(np.float64)
    img_4 = img_4.astype(np.float64)
    
    img_1_masked = np.multiply(img_1,img_mask)
    img_2_masked = np.multiply(img_2,img_mask)
    img_3_masked = np.multiply(img_3,img_mask)
    img_4_masked = np.multiply(img_4,img_mask)
    
    img_1_masked = img_1_masked.astype(np.uint8)
    img_2_masked = img_2_masked.astype(np.uint8)
    img_3_masked = img_3_masked.astype(np.uint8)
    img_4_masked = img_4_masked.astype(np.uint8)
    
    img_2_masked = ndimage.rotate(img_2_masked,90)
    img_3_masked = ndimage.rotate(img_3_masked,180)
    img_4_masked = ndimage.rotate(img_4_masked,270)
    
    temp_img_1 = cv2.add(img_1_masked,img_3_masked)
    temp_img_2 = cv2.add(img_2_masked,img_4_masked)
    
    img_final = cv2.addWeighted(temp_img_1,0.5,temp_img_2,0.5,0)
    
    img_save = np.concatenate((img_1.astype(np.uint8),img_final),axis=1)
    img_save = np.concatenate((img_save,cv2.cvtColor(occp_final,cv2.COLOR_GRAY2BGR)),axis=1)
    
    cv2.imwrite("D:\\driveME\\test\\driveME_video_PNG\\occp\\"+str(i)+".png",img_save)
    
#    try:
#        cv2.imshow("camera 1",img_1.astype(np.uint8))
#        cv2.imshow("final occupancy",occp_final)
#        cv2.imshow("raw image stitch",img_final)
#        cv2.waitKey(0)
#        cv2.destroyAllWindows() 
#    except KeyboardInterrupt:
#        cv2.destroyAllWindows()
#        break