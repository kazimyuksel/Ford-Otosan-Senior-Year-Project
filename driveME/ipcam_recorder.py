# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 16:29:17 2019

@author: Kazım
"""

"""
Continuously capture images from a IPCAM and write to a Redis store.
Usage:
   python recorder.py [width] [height]
"""

import os
from io import BytesIO
import time

import coils
import cv2
import numpy as np
import redis
import argparse
import pickle

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cameraid", required=True, help="Camera ID")
ap.add_argument("-u", "--username", required=True, help="Camera User Name")
ap.add_argument("-p", "--password", required=True, help="Camera Password")
ap.add_argument("-i", "--cameraip", required=True, help="Camera IP Adress")
ap.add_argument("-s", "--serverip", required=True, help="Camera IP Adress")
args = vars(ap.parse_args())

def importHomology(cameraID):
    f = open("camera_calibration_ground_"+str(cameraID)+".driveme", "rb")
    roi_ground = pickle.load(f)
    h_ground = pickle.load(f)
    f.close()
    time.sleep(1)

    return h_ground,roi_ground


# Create video capture object, retrying until successful.
max_sleep = 5.0
cur_sleep = 0.1
camera_address = "rtsp://"+args["username"]+":"+args["password"]+"@"+args["cameraip"]+"/Streaming/Channels/2/picture"
while True:
    cap = cv2.VideoCapture(camera_address)
    if cap.isOpened():
        break
    print('not opened, sleeping {}s'.format(cur_sleep))
    time.sleep(cur_sleep)
    if cur_sleep < max_sleep:
        cur_sleep *= 2
        cur_sleep = min(cur_sleep, max_sleep)
        continue
    cur_sleep = 0.1

# Create client to the Redis store.
store = redis.Redis(host=args["serverip"],port=6379)

# Monitor the framerate at 1s, 5s, 10s intervals.
fps = coils.RateTicker((1, 5, 10))
PORT_DIM = (492,492)
DIM = (640,480)
K=np.array([[344.8127130210307, 0.0, 336.40818490780066], [0.0, 464.63383194201896, 241.32850767967614], [0.0, 0.0, 1.0]])
D=np.array([[0.056114758269676296], [-0.2394852281183658], [0.3620424153302755], [-0.18229098502853122]])

dim1 = DIM
dim2 = (640, 480)
dim3 = (640, 480)
balance=1.0
scaled_K = K * dim1[0] / DIM[0]  # The values of K is to scale with image dimension.
scaled_K[2][2] = 1.0  # Except that K[2][2] is always 1.0
# This is how scaled_K, dim2 and balance are used to determine the final K used to un-distort image. OpenCV document failed to make this clear!
new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, D, dim2, np.eye(3), balance=balance)
map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, D, np.eye(3), new_K, dim3, cv2.CV_16SC2)

# Repeatedly capture current image,
# encode, serialize and push to Redis database.
# Then create unique ID, and push to database as well.
image_key = "image_"+args["cameraid"]
image_id_key = "image_id_"+args["cameraid"]
fps_key = "fps_"+args["cameraid"]
h_ground,roi_ground = importHomology(args["cameraid"])
while True:
    #time.sleep(0.2)
    _, image = cap.read()
    if image is None:
        time.sleep(0.05)
        continue
    image = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    image = cv2.warpPerspective( image, h_ground, PORT_DIM )
    #image = image[PORT_DIM[0]-roi_ground:,:,:]
    _, image = cv2.imencode('.png', image)

    sio = BytesIO()
    np.save(sio, image)
    value = sio.getvalue()
    store.set(image_key, value)
    image_id = os.urandom(8)
    store.set(image_id_key, image_id)
    
    sio_1 = BytesIO() 
    np.save(sio_1, fps.tick()[0])
    value_fps = sio_1.getvalue()
    
    store.set(fps_key, value_fps)

