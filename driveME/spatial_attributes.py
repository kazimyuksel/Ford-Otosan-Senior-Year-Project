# -*- coding: utf-8 -*-
"""
Created on Mon May 13 18:54:49 2019

@author: Kazım
"""
import argparse
import redis
import time
import cv2
from cv2 import aruco
import numpy as np
from io import BytesIO
import pickle
from modules.Aruco import Aruco
import imutils
import os
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--computerID", required=True, help="ID of the current computer.")
ap.add_argument("-t", "--connectionType", required=True, help="Connection type of computer. wifi or eth")
ap.add_argument("-c", "--cameraID", required=True, help="Camera ID to be used.")
args = vars(ap.parse_args())

running = True
def importHomology():
    f = open("camera_calibration_ground_"+str(cameraID)+".driveme", "rb")
    roi_ground = pickle.load(f)
    h_ground = pickle.load(f)
    f.close()
    time.sleep(1)
    f = open("camera_calibration_trailer_"+str(cameraID)+".driveme", "rb")
    roi_trailer = pickle.load(f)
    h_trailer = pickle.load(f)
    f.close()
    time.sleep(1)
    f = open("camera_calibration_relative_"+str(cameraID)+".driveme", "rb")
    h_relative = pickle.load(f)
    f.close()
    return h_ground,roi_ground,h_trailer,roi_trailer,h_relative


aruco_dict = aruco.Dictionary_get( aruco.DICT_4X4_1000 )
arucoParams = aruco.DetectorParameters_create()
def detectAruco(gray_img):
    corners, ids, _= aruco.detectMarkers(gray_img, aruco_dict, parameters=arucoParams)
    return corners, ids

prev_image_id = None
def getImage(camera_id,prev_id=prev_image_id):
    MAX_FPS = 50
    redis_image_key = "image_"+str(camera_id)
    redis_image_id_key = "image_id_"+str(camera_id)
    while True:
        time.sleep(1./MAX_FPS)
        image_id = store.get(redis_image_id_key)
        if image_id != prev_id:
            break
    prev_image_id = image_id
    image = store.get(redis_image_key)
    image = BytesIO(image)
    image = np.load(image)
    img = cv2.imdecode(image, -1)
    return prev_image_id, img, cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

def detectSpecificMarkers(ids):
    if ids is None:
        return None,None
    ids_list = np.ndarray.tolist(ids)
    ids_array = np.array(ids_list)
    ids_array = ids_array.T
    ids_array = ids_array[0,:]
    trailer_index = np.where(ids_array == trailer_id)
    trailer_index = trailer_index[0]
    head_index = np.where(ids_array == head_id)
    head_index = head_index[0]
    if len(head_index) == 0:
        head_index = None

    if len(trailer_index) == 0:
        trailer_index = None

    return head_index, trailer_index

def TrailerMarkerSpatialParameters(corners,marker_index):
    trailer_index = marker_index
    trailer_scalar_index = np.asscalar(trailer_index)
    trailer_corner = corners[trailer_scalar_index]
    pts_trailer = np.array( trailer_corner, np.float32 )
    global_trailer = cv2.perspectiveTransform(pts_trailer,h_relative)
    global_trailer_point = np.array([global_trailer.mean(axis=1)])
    global_trailer_point = global_trailer_point.astype(np.float64)
    pts_trailer = global_trailer[0,:,:]
    orientation_trailer = np.mod((-np.arctan2((pts_trailer[0,1]-pts_trailer[3,1]),(pts_trailer[0,0]-pts_trailer[3,0]))+2*np.pi),2*np.pi)
    global_trailer_point = global_trailer_point[0,:,:]
    global_trailer_point[0,1] = port_shape[0] - global_trailer_point[0,1]
    return (global_trailer_point[0,0], global_trailer_point[0,1]), orientation_trailer
    
    
    
def HeadMarkerSpatialParameters(img,corners,marker_index):
    marker_length = 80
    head_index = marker_index
    head_scalar_index = np.asscalar(head_index)
    head_corner = corners[head_scalar_index]
    pts_head = np.array( head_corner, np.float32 )
    global_head_point = cv2.perspectiveTransform(pts_head.copy(),h_relative)
    global_head_point = np.array([global_head_point.mean(axis=1)])
    global_head_point = global_head_point.astype(np.float64)
    global_head_point = global_head_point[0,:,:]
    global_head_point[0,1] = port_shape[0] - global_head_point[0,1]
    pts_head = np.array([pts_head.mean(axis=1)])
    pts_head = pts_head.astype(np.float64)

    pts_head = pts_head[0,:,:]

    img_rgb = np.pad(img, ((int(marker_length/2.0),int(marker_length/2.0)),(int(marker_length/2.0),int(marker_length/2.0)),(0,0)),"constant",constant_values=((0,0),(0,0),(0,0)))
    head_marker_window = img_rgb[int(pts_head[0,1]):int(pts_head[0,1])+int(marker_length),int(pts_head[0,0]):int(pts_head[0,0])+int(marker_length),:]
    
    head_marker_window_hsv = cv2.cvtColor(head_marker_window,cv2.COLOR_BGR2HSV)
    greenLower = np.array([45, 50, 50])
    greenUpper = np.array([75, 255, 255])
    smooth = cv2.bilateralFilter(head_marker_window_hsv, 3,10, 10)
    mask = cv2.inRange(smooth, greenLower, greenUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
#    cv2.imshow("a",smooth)
#    cv2.imshow("c",mask)
    x,y,radius,pts_marker = None,None,None,None
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
            pts_marker = np.zeros((1,1,2),dtype=np.float64)
            pts_marker[0,0,0] = pts_head[0,0] + x - marker_length/2.0
            pts_marker[0,0,1] = pts_head[0,1] + y - marker_length/2.0 
            global_marker_point = cv2.perspectiveTransform(pts_marker,h_relative)
            global_marker_point = global_marker_point[0,:,:]
            global_marker_point[0,1] = port_shape[0] - global_marker_point[0,1]
    if pts_marker is not None:
        del_yx = (global_marker_point[0,1]-global_head_point[0,1],global_marker_point[0,0]-global_head_point[0,0])
        orientation_head = np.mod((np.arctan2(del_yx[0],del_yx[1])+2*np.pi),2*np.pi)
        return (global_head_point[0,0], global_head_point[0,1]), orientation_head, True
    else:
        return (global_head_point[0,0], global_head_point[0,1]), None, False
        
    
def correctOrientation(camera_id,port_shape,h_pos,h_angle,t_pos,t_angle):
    new_h_pos = None
    new_t_angle = None
    new_h_angle = None
    new_t_pos = None
    if str(camera_id) == "1":
        new_h_pos = h_pos
        new_t_angle = t_angle
        new_h_angle = h_angle
        new_t_pos = t_pos
        
    elif str(camera_id) == "2":
        if h_pos[0] is None or h_pos[1] is None:
            new_h_pos = h_pos
        else:
            theta = np.pi/2.0
            xr = np.cos(theta)*(h_pos[0]-port_shape[1]/2.0)-np.sin(theta)*(h_pos[1]-port_shape[0]/2.0) + port_shape[1]/2.0
            yr = np.sin(theta)*(h_pos[0]-port_shape[1]/2.0)+np.cos(theta)*(h_pos[1]-port_shape[0]/2.0) + port_shape[0]/2.0
            new_h_pos = (xr,yr)
        if t_pos[0] is None or t_pos[1] is None:
            new_t_pos = t_pos
        else:
            theta = np.pi/2.0
            xr = np.cos(theta)*(t_pos[0]-port_shape[1]/2.0)-np.sin(theta)*(t_pos[1]-port_shape[0]/2.0) + port_shape[1]/2.0
            yr = np.sin(theta)*(t_pos[0]-port_shape[1]/2.0)+np.cos(theta)*(t_pos[1]-port_shape[0]/2.0) + port_shape[0]/2.0
            new_t_pos = (xr,yr)
        if h_angle is None:
            new_h_angle = h_angle
        else:
            new_h_angle = h_angle + np.pi/2.0   
            
        if t_angle is None:
            new_t_angle = t_angle
        else:
            new_t_angle = t_angle + np.pi/2.0
    elif str(camera_id) == "3":
        if h_pos[0] is None or h_pos[1] is None:
            new_h_pos = h_pos
        else:
            theta = np.pi
            xr = np.cos(theta)*(h_pos[0]-port_shape[1]/2.0)-np.sin(theta)*(h_pos[1]-port_shape[0]/2.0) + port_shape[1]/2.0
            yr = np.sin(theta)*(h_pos[0]-port_shape[1]/2.0)+np.cos(theta)*(h_pos[1]-port_shape[0]/2.0) + port_shape[0]/2.0
            new_h_pos = (xr,yr)
        if t_pos[0] is None or t_pos[1] is None:
            new_t_pos = t_pos
        else:
            theta = np.pi
            xr = np.cos(theta)*(t_pos[0]-port_shape[1]/2.0)-np.sin(theta)*(t_pos[1]-port_shape[0]/2.0) + port_shape[1]/2.0
            yr = np.sin(theta)*(t_pos[0]-port_shape[1]/2.0)+np.cos(theta)*(t_pos[1]-port_shape[0]/2.0) + port_shape[0]/2.0
            new_t_pos = (xr,yr)
        if h_angle is None:
            new_h_angle = h_angle
        else:
            new_h_angle = h_angle + np.pi  
        if t_angle is None:
            new_t_angle = t_angle
        else:
            new_t_angle = t_angle + np.pi
        
    elif str(camera_id) == "4":
        if h_pos[0] is None or h_pos[1] is None:
            new_h_pos = h_pos
        else:
            theta = np.pi*(3.0/2.0)
            xr = np.cos(theta)*(h_pos[0]-port_shape[1]/2.0)-np.sin(theta)*(h_pos[1]-port_shape[0]/2.0) + port_shape[1]/2.0
            yr = np.sin(theta)*(h_pos[0]-port_shape[1]/2.0)+np.cos(theta)*(h_pos[1]-port_shape[0]/2.0) + port_shape[0]/2.0
            new_h_pos = (xr,yr)
        if t_pos[0] is None or t_pos[1] is None:
            new_t_pos = t_pos
        else:
            theta = np.pi*(3.0/2.0)
            xr = np.cos(theta)*(t_pos[0]-port_shape[1]/2.0)-np.sin(theta)*(t_pos[1]-port_shape[0]/2.0) + port_shape[1]/2.0
            yr = np.sin(theta)*(t_pos[0]-port_shape[1]/2.0)+np.cos(theta)*(t_pos[1]-port_shape[0]/2.0) + port_shape[0]/2.0
            new_t_pos = (xr,yr)
        if h_angle is None:
            new_h_angle = h_angle
        else:
            new_h_angle = h_angle + np.pi*(3.0/2.0)
        if t_angle is None:
            new_t_angle = t_angle
        else:
            new_t_angle = t_angle + np.pi*(3.0/2.0)
    else:
        raise ValueError("No camera exist like that.")
    if h_angle is not None:
        new_h_angle = np.mod(new_h_angle+2*np.pi,2*np.pi)
    if t_angle is not None:
        new_t_angle = np.mod(new_t_angle+2*np.pi,2*np.pi)
    return new_h_pos, new_h_angle, new_t_pos, new_t_angle


cameraID = args["cameraID"]
hostID = args["computerID"]
connectionType = args["connectionType"]
if connectionType == "eth":
    hostIP = "192.168.1.10"+hostID
else:
    hostIP = "192.168.1.20"+hostID
store = redis.Redis(host=hostIP,port=6379)
head = Aruco()
trailer = Aruco()

head_id = 5
trailer_id = 2
port_shape = (492,492)

h_ground,roi_ground,h_trailer,roi_trailer,h_relative = importHomology()
dt = 0.05
elapsed_time = 0.02
start_time = time.time()
try:
    head.resetParameters
    trailer.resetParameters
    while running:

        dt = 0.9*dt + 0.1*elapsed_time
        head_exist = True
        trailer_exist = True

        prev_image_id,img_color,img_gray = getImage(cameraID,prev_id=prev_image_id)
        corners, ids = detectAruco(img_gray)
        head_index,trailer_index = detectSpecificMarkers(ids)
        if head_index is None:
            head_exist = False
   
        if trailer_index is None:
            trailer_exist = False

        if trailer_exist:
            trailerPos, trailerAngle = TrailerMarkerSpatialParameters(corners,trailer_index)
            trailer.setPos(trailerPos)
            trailer.setAngle(trailerAngle)
        else:
            trailer.resetParameters
                
        if head_exist:
            headPos,headAngle,isMarkerExist= HeadMarkerSpatialParameters(img_color,corners,head_index)
            head.setPos(headPos)
            head.setAngle(headAngle)
        else:
            head.resetParameters
        new_h_pos, new_h_angle, new_t_pos, new_t_angle = correctOrientation(cameraID,port_shape,head.Pos,head.Angle,trailer.Pos,trailer.Angle)
        output = "Code Hz: {:.2f}\t ,\t h_x: {:.2f}\t ,\t h_y: {:.2f}\t ,\t h_angle: {:.2f}\t ,\t t_x: {:.2f}\t ,\t t_y: {:.2f}\t ,\t t_angle: {:.2f}".format(np.float64(1.0/dt),np.float64(new_h_pos[0]),np.float64(new_h_pos[1]),np.float64(new_h_angle),np.float64(new_t_pos[0]),np.float64(new_t_pos[1]),np.float64(new_t_angle))
        print(output)
        redis_data_write = np.append(new_h_pos, new_h_angle)
        redis_data_write = np.append(redis_data_write, new_t_pos)
        redis_data_write = np.append(redis_data_write, new_t_angle)

        sio = BytesIO() 
        np.save(sio, redis_data_write)
        value = sio.getvalue()

        store.set("cam_data_"+str(cameraID), value)
        position_data_id = os.urandom(8)
        store.set("cam_data_id_"+str(cameraID), position_data_id)

        elapsed_time = time.time() - start_time
        start_time =time.time()

except (KeyboardInterrupt):      
    pass
    


