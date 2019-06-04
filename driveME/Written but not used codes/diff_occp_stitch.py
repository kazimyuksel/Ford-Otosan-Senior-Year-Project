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
ground_1 = cv2.imread("D:/driveME/test/kamera_1.png",1)
ground_2 = cv2.imread("D:/driveME/test/kamera_2.png",1)
ground_3 = cv2.imread("D:/driveME/test/kamera_3.png",1)
ground_4 = cv2.imread("D:/driveME/test/kamera_4.png",1)
counter = 0
kernel = np.ones((5,5),np.uint8)

port_shape = (492,492)
center = int(port_shape[1]/2)
mask_width = 40.0
img_mask = np.ones((492,492),dtype=np.float64)
img_mask[:int(center+mask_width/2),:] = 0.0
mask_step = 1.0/mask_width
for j in range(int(mask_width)):
    start_pos = int(center-mask_width/2)
    img_mask[int(j+start_pos),:] = j*mask_step
#img_mask = cv2.resize(img_mask,(250,250))

try:
    while True:
        
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
        
        dif_1 = np.subtract(ground_1.copy(),img_1)
        dif_1 = (dif_1-dif_1.min())/(dif_1.max()-dif_1.min())
        dif_1 = 255.0*dif_1
        dif_1 = cv2.cvtColor(dif_1.astype(np.uint8),cv2.COLOR_BGR2GRAY)
#        dif_1 = np.power(dif_1,1.3)
#        dif_1 = (dif_1-dif_1.min())/(dif_1.max()-dif_1.min())
#        dif_1 = 255.0*dif_1

#        _,thresh_1 = cv2.threshold(dif_1,40,255,cv2.THRESH_BINARY)
#        thresh_1 =cv2.erode(thresh_1,kernel,iterations = 1)
#        thresh_1 =cv2.dilate(thresh_1,kernel,iterations = 6)
#        thresh_1 =cv2.erode(thresh_1,kernel,iterations = 6)

        dif_2 = np.subtract(ground_2.copy(),img_2)
        dif_2 = (dif_2-dif_2.min())/(dif_2.max()-dif_2.min())
        dif_2 = 255.0*dif_2
        dif_2 = cv2.cvtColor(dif_2.astype(np.uint8),cv2.COLOR_BGR2GRAY)
#        dif_2 = np.power(dif_2,1.3)
#        dif_2 = (dif_2-dif_2.min())/(dif_2.max()-dif_2.min())
#        dif_2 = 255.0*dif_2

#        _,thresh_2 = cv2.threshold(dif_2,40,255,cv2.THRESH_BINARY)
#        thresh_2 =cv2.erode(thresh_2,kernel,iterations = 1)
#        thresh_2 =cv2.dilate(thresh_2,kernel,iterations = 6)
#        thresh_2 =cv2.erode(thresh_2,kernel,iterations = 6)

        dif_3 = np.subtract(ground_3.copy(),img_3)
        dif_3 = (dif_3-dif_3.min())/(dif_3.max()-dif_3.min())
        dif_3 = 255.0*dif_3
        dif_3 = cv2.cvtColor(dif_3.astype(np.uint8),cv2.COLOR_BGR2GRAY)
#        dif_3 = np.power(dif_3,1.3)
#        dif_3 = (dif_3-dif_3.min())/(dif_3.max()-dif_3.min())
#        dif_3 = 255.0*dif_3

#        _,thresh_3 = cv2.threshold(dif_3,40,255,cv2.THRESH_BINARY)
#        thresh_3 =cv2.erode(thresh_3,kernel,iterations = 1)
#        thresh_3 =cv2.dilate(thresh_3,kernel,iterations = 6)
#        thresh_3 =cv2.erode(thresh_3,kernel,iterations = 6)
#        
        
        dif_4 = np.subtract(ground_4.copy(),img_4)
        dif_4 = (dif_4-dif_4.min())/(dif_4.max()-dif_4.min())
        dif_4 = 255.0*dif_4
        dif_4 = cv2.cvtColor(dif_4.astype(np.uint8),cv2.COLOR_BGR2GRAY)
#        dif_4 = np.power(dif_4,1.3)
#        dif_4 = (dif_4-dif_4.min())/(dif_4.max()-dif_4.min())
#        dif_4 = 255.0*dif_4

        _,thresh_4 = cv2.threshold(dif_4,40,255,cv2.THRESH_BINARY)
#        thresh_4 =cv2.erode(thresh_4,kernel,iterations = 1)
#        thresh_4 =cv2.dilate(thresh_4,kernel,iterations = 6)
#        thresh_4 =cv2.erode(thresh_4,kernel,iterations = 6)
#        
#        img_1 = thresh_1
#        img_2 = thresh_2
#        img_3 = thresh_3
#        img_4 = thresh_4
#        
        img_1 = dif_1
        img_2 = dif_2
        img_3 = dif_3
        img_4 = dif_4
         
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
        cv2.imshow("a",img_final)
        if cv2.waitKey(27) & 0xFF == ord('q') :
            break
    cv2.destroyAllWindows()

except KeyboardInterrupt:
    cv2.destroyAllWindows()