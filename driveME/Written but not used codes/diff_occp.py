import redis
import time
import numpy as np
from io import BytesIO
import cv2
MAX_FPS = 50
prev_image_id = None
prev_position_data_id = None
store = redis.Redis(host="192.168.1.101",port=6379)
ground = cv2.imread("D:/driveME/test/kamera_3.png",1)
counter = 0
kernel = np.ones((5,5),np.uint8)
try:
    while True:
        while True:
            time.sleep(1./MAX_FPS)
            image_id = store.get('image_id_3')
            if image_id != prev_image_id:
                break
        prev_image_id = image_id
        image = store.get('image_3')
        image = BytesIO(image)
        image = np.load(image)
        rgb_1 = cv2.imdecode(image, -1)
        
        a = np.subtract(ground.copy(),rgb_1)
        a = cv2.cvtColor(a.astype(np.uint8),cv2.COLOR_BGR2GRAY)
        a = (a-a.min())/(a.max()-a.min())
        a = 255.0*a
        a = cv2.cvtColor(a= a.astype(np.uint8),cv2.COLOR_BGR2GRAY)
#        a = np.power(a,1.3)

        
        
        _,thres = cv2.threshold(a,30,255,cv2.THRESH_BINARY)
        thres =cv2.erode(thres,kernel,iterations = 1)
        thres =cv2.dilate(thres,kernel,iterations = 6)
        thres =cv2.erode(thres,kernel,iterations = 6)
        cv2.imwrite("D:/driveME/subst.png",a.astype(np.uint8))
except KeyboardInterrupt:
    cv2.destroyAllWindows()