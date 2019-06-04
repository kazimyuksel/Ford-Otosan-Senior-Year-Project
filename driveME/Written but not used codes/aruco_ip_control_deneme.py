
from __future__ import division


import time

import cv2
assert cv2.__version__[0] >= '3', 'The program requires OpenCV version greater than >= 3.0.0'
from cv2 import aruco
import numpy as np
from io import BytesIO
import redis
import os
import time
import h5py
def draw(img, corners, imgpts):
    corner = tuple(corners[0][0][1].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()),(0,255,0), 2)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()),(255,0,0), 2)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()),(0,0,255), 2)
    return img

width = 640
height = 480
scaling_ratio = 1
new_width = scaling_ratio*width
new_width = (int)(new_width)
new_height = scaling_ratio*height
new_height = (int)(new_height)

DIM = (640,480)
K=np.array([[344.8127130210307, 0.0, 336.40818490780066], [0.0, 464.63383194201896, 241.32850767967614], [0.0, 0.0, 1.0]])
D=np.array([[0.056114758269676296], [-0.2394852281183658], [0.3620424153302755], [-0.18229098502853122]])

dim1 = DIM
dim2 = (640, 480)
dim3 = (640, 480)
balance=1.0

bool_start_1 = True
bool_start_2 = True

scaled_K = K * dim1[0] / DIM[0]  # The values of K is to scale with image dimension.
scaled_K[2][2] = 1.0  # Except that K[2][2] is always 1.0
# This is how scaled_K, dim2 and balance are used to determine the final K used to un-distort image. OpenCV document failed to make this clear!
new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, D, dim2, np.eye(3), balance=balance)
map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, D, np.eye(3), new_K, dim3, cv2.CV_16SC2)

aruco_dict = aruco.Dictionary_get( aruco.DICT_4X4_1000 )
markerLength = 35.0
arucoParams = aruco.DetectorParameters_create()

store = redis.Redis(host='192.168.1.61',port=6379)
prev_image_id = None

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-12)
counter = 1
MAX_FPS = 50

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-12)
corners_array = np.empty((0,4,2))
start_time = time.time()
avg_hz = 0


global_head_points = np.array([], dtype=np.int32).reshape(0,2)

with h5py.File('D://driveME_IPCAM1//Calibration//IPCAM1_TRAILER_TOP_CALIBRATION.h5', 'r') as hf:
    h = hf['trailer_top_calib'][:]
trailer_index = None
while True:
    while True:
        time.sleep(1./MAX_FPS)
        image_id = store.get('image_id')
        if image_id != prev_image_id:
            break
    prev_image_id = image_id
    image = store.get('image')
    image = BytesIO(image)
    image = np.load(image)
    rgb = cv2.imdecode(image, -1)
    #rgb = cv2.remap(rgb, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    
    img_smooth = cv2.bilateralFilter(rgb, 5,75, 75)
    gray = cv2.cvtColor(img_smooth, cv2.COLOR_BGR2GRAY) 
    img_orj = img_smooth
    img_smooth = cv2.resize(img_smooth,(new_width,new_height)) 
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=arucoParams)
    out = cv2.warpPerspective( rgb, h, ( 492, 492 ) )
   
     
    if ids is None:
        calibration_index = []
        calibration_index = np.array(calibration_index)
        head_index = []
        head_index = np.array(head_index).astype(np.int32)
        imgWithAruco = rgb
        #calibration_index.size
    else:
        
        imgWithAruco = aruco.drawDetectedMarkers(rgb, corners, ids, (0,255,0))
        ids_list = np.ndarray.tolist(ids)
        ids_array = np.array(ids_list)
        ids_array = ids_array.T
        ids_array = ids_array[0,:]
        trailer_index = np.where(ids_array == 2)
        trailer_index = trailer_index[0]
        if trailer_index.size == 1:
            trailer_scalar_index = np.asscalar(trailer_index)
            trailer_corner = corners[trailer_scalar_index]
            pts_head = np.array( trailer_corner, np.float32 )
            global_head = cv2.perspectiveTransform(pts_head,h)
            global_head_point = np.array([global_head.mean(axis=1)])
            global_head_point = global_head_point.astype(np.float64)
            global_head_point_as_int = global_head_point.astype(np.int32)
            global_head_point_as_int = global_head_point_as_int[0,:,:]
            global_head_points = np.vstack([global_head_points, global_head_point_as_int])
            pts = global_head[0,:,:]
            orientation = np.mod((-np.arctan2((pts[0,1]-pts[3,1]),(pts[0,0]-pts[3,0]))+2*np.pi),2*np.pi)
            global_head_point = global_head_point[0,:,:]
            global_head_point[0,1] = out.shape[0] - global_head_point[0,1]
            redis_data_write = np.append(global_head_point.astype(np.float64), orientation)
            sio = BytesIO() 
            np.save(sio, redis_data_write)
            value = sio.getvalue()
            
           
            store.set('position_data', value)
            position_data_id = os.urandom(4)
            store.set('position_data_id', position_data_id)
            
            
            # LOOP DIŞINDA INIT KISMI
            import redis
            import os
            import time
            import numpy as np
            from io import BytesIO
            store = redis.Redis(host='192.168.1.61',port=6379)
            prev_position_data_id = None
            
            # CONTROL LOOPUNDA DATA ISTEDIGIN YERE
            while True:
                time.sleep(0.05)
                position_data_id = store.get('position_data_id')
                if position_data_id != prev_position_data_id:
                    break
            prev_position_data_id = position_data_id
            byte_get = store.get('position_data')
            byte_get = BytesIO(byte_get)
            float_get = np.load(byte_get)

            

    cv2.imshow('occupancy', imgWithAruco)
    if trailer_index.size == 1:
        out = cv2.polylines(out,[global_head_points],False,(0,255,0),2)
    cv2.imshow('bird', out)
    elapsed_time = time.time() - start_time
    start_time =time.time()
    avg_hz = 0.8*avg_hz + 0.2/elapsed_time
    #print(avg_hz)
    if cv2.waitKey(27) & 0xFF == ord('q') :
        break
    
cv2.destroyAllWindows()



