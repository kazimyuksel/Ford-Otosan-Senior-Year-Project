# -*- coding: utf-8 -*-
"""
Created on Fri May 24 20:04:01 2019

@author: Kazım
"""

# -*- coding: utf-8 -*-
"""
Created on Tue May 14 10:01:20 2019

@author: Kazım
"""
import redis
import time
import numpy as np
from io import BytesIO
import cv2
from scipy import ndimage
MAX_FPS = 50
prev_image_id = None
prev_position_data_id = None
store = redis.Redis(host="192.168.1.101",port=6379)

port_shape = (492,492)
center = int(port_shape[1]/2)
mask_width = 40.0
img_mask = np.ones((492,492),dtype=np.float64)
img_mask[:int(center+mask_width/2),:] = 0.0
mask_step = 1.0/mask_width
for j in range(int(mask_width)):
    start_pos = int(center-mask_width/2)
    img_mask[int(j+start_pos),:] = j*mask_step
img_mask = cv2.resize(img_mask,(300,300))
try:
    while True:
        img_1 = np.zeros((492,492,3),dtype=np.uint8)
        while True:
            time.sleep(1./MAX_FPS)
            image_id = store.get('image_id_1')
            if image_id != prev_image_id:
                break
        prev_image_id = image_id
        image = store.get('image_1')
        image = BytesIO(image)
        image = np.load(image)
        img_1 = cv2.imdecode(image, -1)
        img_2 = np.zeros((492,492,3),dtype=np.uint8)
        while True:
            time.sleep(1./MAX_FPS)
            image_id = store.get('image_id_2')
            if image_id != prev_image_id:
                break
        prev_image_id = image_id
        image = store.get('image_2')
        image = BytesIO(image)
        image = np.load(image)
        img_2 = cv2.imdecode(image, -1)
        img_3 = np.zeros((492,492,3),dtype=np.uint8)
        while True:
            time.sleep(1./MAX_FPS)
            image_id = store.get('image_id_3')
            if image_id != prev_image_id:
                break
        prev_image_id = image_id
        image = store.get('image_3')
        image = BytesIO(image)
        image = np.load(image)
        img_3 = cv2.imdecode(image, -1)
        img_4 = np.zeros((492,492,3),dtype=np.uint8)
        while True:
            time.sleep(1./MAX_FPS)
            image_id = store.get('image_id_4')
            if image_id != prev_image_id:
                break
        prev_image_id = image_id
        image = store.get('image_4')
        image = BytesIO(image)
        image = np.load(image)
        img_4 = cv2.imdecode(image, -1)
        
        img_1 = cv2.cvtColor(img_1,cv2.COLOR_BGR2GRAY)
        img_2 = cv2.cvtColor(img_2,cv2.COLOR_BGR2GRAY)
        img_3 = cv2.cvtColor(img_3,cv2.COLOR_BGR2GRAY)
        img_4 = cv2.cvtColor(img_4,cv2.COLOR_BGR2GRAY)
        
        
        img_1 = cv2.resize(img_1,(300,300))
        img_2 = cv2.resize(img_2,(300,300))
        img_3 = cv2.resize(img_3,(300,300))
        img_4 = cv2.resize(img_4,(300,300))
        
        temp_img_1 = np.zeros((300,300))
        temp_img_2 = np.zeros((300,300))
        img_final = np.zeros((300,300))
        
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
#        
#        while True:
#            time.sleep(0.05)
#            position_data_id = store.get('master_spatial_attributes_id')
#            if position_data_id != prev_position_data_id:
#                break
#        prev_position_data_id = position_data_id
#        byte_get = store.get('master_spatial_attributes_data')
#        byte_get = BytesIO(byte_get)
#        float_get = np.load(byte_get)
#        
#        head_x = float_get[0]
#        head_y = float_get[1]
#        head_orientation = float_get[2]
#        trailer_x = float_get[3]
#        trailer_y = float_get[4]
#        trailer_orientation = float_get[5]
#        length= 40.0
#        if not np.isnan(head_x) and not np.isnan(head_orientation):
#            x2 = head_x + length * np.cos(head_orientation)
#            y2 = 492-head_y - length * np.sin(head_orientation)
#            rgb=cv2.circle(rgb,(int(head_x),int(492-head_y)), 5, (0,0,255), -1)
#            rgb = cv2.line(rgb, (int(head_x),int(492-head_y)), (int(x2),int(y2)), (0,0,255), 3) 
#        if not np.isnan(trailer_x) and not np.isnan(trailer_orientation):
#            x2 = trailer_x + length * np.cos(trailer_orientation)
#            y2 = 492-trailer_y - length * np.sin(trailer_orientation)
#            rgb = cv2.line(rgb, (int(trailer_x),int(492-trailer_y)), (int(x2),int(y2)), (0,255,), 3) 
#            rgb=cv2.circle(rgb,(int(trailer_x),int(492-trailer_y)), 5, (0,255,0), -1)

        cv2.imshow("a",img_final)
        if cv2.waitKey(27) & 0xFF == ord('q') :
            break
    cv2.destroyAllWindows()
except KeyboardInterrupt:
    cv2.destroyAllWindows()
