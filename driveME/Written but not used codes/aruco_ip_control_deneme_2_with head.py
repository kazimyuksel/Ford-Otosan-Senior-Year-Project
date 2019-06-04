
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
import imutils

trailer_wheel_base = 56.1
head_wheel_base = 27.0
head_track_width = 13.395
fifth_wheel_distance = 5.25

delay = 400 #ms
trailer_index = None
head_index= None
head_speed = 0
firstLoop = True
dt = 10
steer_angle = 0*np.pi/180

prev_steer_id = None



x_path = "D:/Python Controller/Paths/file_x_2.txt"
y_path = "D:/Python Controller/Paths/file_y_2.txt"
obstacles_x = "D:/Python Controller/Paths/xCoordinateOfObstacles.txt"
obstacles_y = "D:/Python Controller/Paths/yCoordinateOfObstacles.txt"
      
def get_path(path_x, path_y, obstacles_x, obstacles_y):
    
    ### Load coordinates
    path_x = np.loadtxt(path_x)
    path_y = np.loadtxt(path_y)
    obstacles_x = np.loadtxt(obstacles_x)
    obstacles_y = np.loadtxt(obstacles_y)
    
    map_x_coord = np.zeros((1,1))
    map_y_coord = np.zeros((1,1))
    
    path_x = path_x.reshape((1,len(path_x)))
    path_y = path_y.reshape((1,len(path_y)))
    
    path_x = np.transpose(path_x)
    path_y = np.transpose(path_y)
    
    ### Increase number of points
    for i in range(len(path_x) - 1):
        x_concat = np.linspace(path_x[i], path_x[i+1], 1)
        y_concat = np.linspace(path_y[i], path_y[i+1], 1)
        
        x_concat = x_concat.reshape((1,1))
        y_concat = y_concat.reshape((1,1))
        
        x_concat = np.transpose(x_concat)
        y_concat = np.transpose(y_concat)
        
        map_x_coord = np.concatenate((map_x_coord, x_concat), axis = 0)
        map_y_coord = np.concatenate((map_y_coord, y_concat), axis = 0)
        
    map_x_coord = map_x_coord[1:]
    map_y_coord = map_y_coord[1:]
    
    obstacles_x = obstacles_x.reshape((1,len(obstacles_x)))
    obstacles_y = obstacles_y.reshape((1,len(obstacles_y)))
    obstacles_x = np.transpose(obstacles_x)
    obstacles_y = np.transpose(obstacles_y)
    obstacles = np.concatenate((obstacles_x, obstacles_y), axis = 1)
    ### Expand Area
    obstacles = obstacles*10
    map_x_coord = map_x_coord*10
    map_y_coord = map_y_coord*10
    reference_path = np.concatenate((map_x_coord, map_y_coord), axis = 1)
    
    return map_x_coord,map_y_coord, obstacles
x_coord, y_coord, b = get_path(x_path, y_path, obstacles_x, obstacles_y)


coord = np.array([x_coord[:,0], 492-y_coord[:,0]]).astype(np.int32).T




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
avg_time = 0

global_trailer_points = np.array([], dtype=np.int32).reshape(0,2)
global_head_points = np.array([], dtype=np.int32).reshape(0,2)
orientation_trailer = 0
orientation_head = 0
global_head_rear_axis_point = np.empty((0,2),dtype=np.float64)
with h5py.File('D://driveME_IPCAM1//Calibration//IPCAM1_TRAILER_TOP_CALIBRATION.h5', 'r') as hf:
    h = hf['trailer_top_calib'][:]








def _kinematicSolver(head_speed, head_angle, trailer_angle, steer_angle, head_position, dt):

    lh = head_wheel_base
    lc = fifth_wheel_distance
    lt = trailer_wheel_base
    
    head_speed_x = head_speed*np.cos(head_angle)
    head_speed_y = head_speed*np.sin(head_angle)
    heading_dot = (head_speed/lh)*np.tan(steer_angle)
    head_new_pos_x = head_position[0,0] + dt*head_speed_x
    head_new_pos_y = head_position[0,1] + dt*head_speed_y
    head_angle_new = head_angle + dt*heading_dot
    
    theta_s = steer_angle
    theta_h = head_angle_new
    theta_t = trailer_angle
    theta_n = theta_t
    Vh = head_speed
    F1 = (Vh/lt)*(np.tan(theta_s)*np.cos(theta_h-theta_t)*lc/lh+np.sin(theta_h-theta_t))
    theta_t = theta_n + 0.25*dt*F1
    F2 = (Vh/lt)*(np.tan(theta_s)*np.cos(theta_h-theta_t)*lc/lh+np.sin(theta_h-theta_t))
    theta_t = theta_n +(3/32)*dt*F1 + (9/32)*dt*F2
    F3 = (Vh/lt)*(np.tan(theta_s)*np.cos(theta_h-theta_t)*lc/lh+np.sin(theta_h-theta_t))
    theta_t = theta_n + (1932/2197)*dt*F1 - (7200/2197)*dt*F2 + (7296/2197)*dt*F3
    F4 = (Vh/lt)*(np.tan(theta_s)*np.cos(theta_h-theta_t)*lc/lh+np.sin(theta_h-theta_t))
    theta_t = theta_n + (439/216)*dt*F1 - 8*dt*F2 + (3680/513)*dt*F3 - (845/4104)*dt*F4
    F5 = (Vh/lt)*(np.tan(theta_s)*np.cos(theta_h-theta_t)*lc/lh+np.sin(theta_h-theta_t))
    theta_t = theta_n - (8/27)*dt*F1 + 2*dt*F2 - (3544/2565)*dt*F3 + (1859/4104)*dt*F4  - (11/40)*dt*F5
    F6 = (Vh/lt)*(np.tan(theta_s)*np.cos(theta_h-theta_t)*lc/lh+np.sin(theta_h-theta_t))
    
    theta_n1 = theta_n + dt*((16/135)*F1+(6656/12825)*F3+(28561/56430)*F4-(9/50)*F5+(2/55)*F6)
    new_theta_t = theta_n1;
    trailer_rear_x = head_new_pos_x + lc*np.cos(theta_h) + lt*np.cos(theta_n1+np.pi)
    trailer_rear_y = head_new_pos_y + lc*np.sin(theta_h) + lt*np.sin(theta_n1+np.pi)
    trailer_new_pos = np.array([trailer_rear_x, trailer_rear_y]).reshape((1,2))
    head_new_pos =  np.array([head_new_pos_x, head_new_pos_y]).reshape((1,2))
    return head_angle_new, new_theta_t, head_new_pos, trailer_new_pos

def forwardPredictor(delay, dt, trailer_angle, head_pos, head_angle, steer_angle, head_speed):
    head = np.empty((0,2),dtype=np.float64)  
    trailer = np.empty((0,2),dtype=np.float64)  

    loop_count = delay/dt
    loop_count = np.round(loop_count)
    new_dt = delay/loop_count
    new_dt = new_dt/1000
    head_angle_loop, trailer_angle_loop, head_pos_loop, trailer_pos_loop = _kinematicSolver(head_speed, head_angle, trailer_angle, steer_angle, head_pos,dt= new_dt) 
    for i in range(int(loop_count)-1):
        head_angle_loop, trailer_angle_loop, head_pos_loop, trailer_pos_loop = _kinematicSolver(head_speed, head_angle_loop, trailer_angle_loop, steer_angle, head_pos_loop,dt= new_dt) 
        conc_h = head_pos_loop.copy()
        conc_t = trailer_pos_loop.copy()
        conc_h[0,1] = out.shape[0] - conc_h[0,1]
        conc_t[0,1] = out.shape[0] - conc_t[0,1]
        head = np.concatenate((head,np.array(conc_h).reshape((1,2))))
        trailer = np.concatenate((trailer,np.array(conc_t).reshape((1,2))))

    return head_angle_loop, trailer_angle_loop, head_pos_loop, trailer_pos_loop,head,trailer



steer_angle =0*np.pi/180

head = np.empty((0,2),dtype=np.float64)  
trailer = np.empty((0,2),dtype=np.float64)  
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
        head_index = np.where(ids_array == 5)
        head_index = head_index[0]
        if trailer_index.size == 1 and head_index.size == 1:
            trailer_scalar_index = np.asscalar(trailer_index)
            trailer_corner = corners[trailer_scalar_index]
            head_scalar_index = np.asscalar(head_index)
            head_corner = corners[head_scalar_index]
            
            pts_trailer = np.array( trailer_corner, np.float32 )
            global_trailer = cv2.perspectiveTransform(pts_trailer,h)
            global_trailer_point = np.array([global_trailer.mean(axis=1)])
            global_trailer_point = global_trailer_point.astype(np.float64)
            global_trailer_point_as_int = global_trailer_point.astype(np.int32)
            global_trailer_point_as_int = global_trailer_point_as_int[0,:,:]
            global_trailer_points = np.vstack([global_trailer_points, global_trailer_point_as_int])
            pts_trailer = global_trailer[0,:,:]
            
            pts_head = np.array( head_corner, np.float32 )
            global_head = cv2.perspectiveTransform(pts_head,h)
            global_head_point = np.array([global_head.mean(axis=1)])
            global_head_point = global_head_point.astype(np.float64)
            image_global_head_point = global_head_point.copy()
            image_global_head_point = image_global_head_point[0,:,:]
            global_head_point_as_int = global_head_point.astype(np.int32)
            global_head_point_as_int = global_head_point_as_int[0,:,:]
            global_head_points = np.vstack([global_head_points, global_head_point_as_int])
            pts_head = global_head[0,:,:]
            
            head_marker_window = out[int(global_head_point[0,0,1])-40:int(global_head_point[0,0,1])+40,int(global_head_point[0,0,0])-40:int(global_head_point[0,0,0])+40]
            greenLower = np.array([45, 50, 50])
            greenUpper = np.array([75, 255, 255])
            blurred = cv2.GaussianBlur(head_marker_window, (5, 5), 0)
            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, greenLower, greenUpper)
            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)
            
            cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            center = None
         
        	# only proceed if at least one contour was found
            if len(cnts) > 0:
                # find the largest contour in the mask, then use
                # it to compute the minimum enclosing circle and
                # centroid
                c = max(cnts, key=cv2.contourArea)
                ((x, y), radius) = cv2.minEnclosingCircle(c)
        		# only proceed if the radius meets a minimum size
                if radius > 2:
        			# draw the circle and centroid on the frame,
        			# then update the list of tracked points
                    cv2.circle(head_marker_window, (int(x), int(y)), int(radius),(255, 0, 255), 1)
                    cv2.line(head_marker_window, (int(x),int(y)) , (40,40),(0, 255, 255), 2)
                    cv2.line(out, (int(global_head_point_as_int[0,0]),int(global_head_point_as_int[0,1])) , (int(global_trailer_point_as_int[0,0]),int(global_trailer_point_as_int[0,1])),(0, 0, 255), 2)
                    #cv2.circle(frame, center, 5, (0, 0, 255), -1)
                    
            cv2.imshow("asd", head_marker_window )
            orientation_head = np.mod((-np.arctan2((y-40),(x-40))+2*np.pi),2*np.pi)
            orientation_trailer = np.mod((-np.arctan2((pts_trailer[0,1]-pts_trailer[3,1]),(pts_trailer[0,0]-pts_trailer[3,0]))+2*np.pi),2*np.pi)
            
            global_trailer_point = global_trailer_point[0,:,:]
            global_trailer_point[0,1] = out.shape[0] - global_trailer_point[0,1]
            
            global_head_point = global_head_point[0,:,:]
            
            global_head_point[0,1] = out.shape[0] - global_head_point[0,1]
            
            global_head_rear_axis_point = image_global_head_point - np.array([fifth_wheel_distance*np.cos(orientation_head),-fifth_wheel_distance*np.sin(orientation_head)],dtype=np.float64)
            #print(global_head_rear_axis_point)
            #print(orientation_head)
            cv2.line(out, (int(global_head_rear_axis_point[0,0]),int(global_head_rear_axis_point[0,1])) , (int(image_global_head_point[0,0]),int(image_global_head_point[0,1])),(0, 255, 0), 2)
            global_head_rear_axis_point[0,1] = out.shape[0] - global_head_rear_axis_point[0,1]
                   
            if not store.exists('steer_id'):
                steer_angle= 0.0
                
            else:
                while True:
                    time.sleep(0.05)
                    steer_id = store.get('steer_id')
                    if steer_id != prev_steer_id:
                        break
                steer_id = steer_id
                byte_get = store.get('steer_data')
                byte_get = BytesIO(byte_get)
                steer_angle = np.load(byte_get)
            if firstLoop:
                firstLoop = False
                temp_head = global_head_rear_axis_point
                del_position = 0
            else:
                del_position = global_head_rear_axis_point - temp_head
                temp_head = global_head_rear_axis_point
                del_position = np.linalg.norm(del_position)
                velocity = del_position / avg_time
                if velocity < 3.0:
                    velocity = 0.0
                head_speed = head_speed*0.7 + velocity*0.3
                print(head_speed)
                head_angle, trailer_angle, head_pos, trailer_pos,head,trailer = forwardPredictor(delay, dt, orientation_trailer, global_head_rear_axis_point, orientation_head, steer_angle, head_speed)
            
                
            loop_trailer_wheelbase = np.linalg.norm(global_head_point-global_trailer_point)
            
            
            
            redis_data_write = np.append(trailer_pos, trailer_angle)
            sio = BytesIO() 
            np.save(sio, redis_data_write)
            value = sio.getvalue()

            store.set('position_data', value)
            position_data_id = os.urandom(4)
            store.set('position_data_id', position_data_id)
            
           

            

    cv2.imshow('occupancy', imgWithAruco)
    if trailer_index.size == 1 and head_index.size == 1:
        out = cv2.polylines(out,[global_trailer_points],False,(0,255,0),2)
        if not firstLoop:
            out = cv2.polylines(out,[trailer.astype(np.int32)],False,(255,0,0),2)
            out = cv2.polylines(out,[head.astype(np.int32)],False,(255,255,0),2)
            out = cv2.polylines(out,[coord],False,(255,255,0),2)
    cv2.imshow('bird', out)
    elapsed_time = time.time() - start_time
    start_time =time.time()
    avg_hz = 0.7*avg_hz + 0.3/elapsed_time
    avg_time = 1.0/avg_hz
    if cv2.waitKey(27) & 0xFF == ord('q') :
        break
    
cv2.destroyAllWindows()



