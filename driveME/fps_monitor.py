# -*- coding: utf-8 -*-
"""
Created on Thu May  2 12:14:30 2019

@author: Kazım
"""
from progressbar import FormatLabel, ProgressBar

import redis
import argparse
import time
from io import BytesIO
import numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cameraID", required=True, help="Camera ID")
ap.add_argument("-s", "--serverIP", required=True, help="Camera IP Adress")
args = vars(ap.parse_args())

camera_id = args["cameraID"].split(",")
store = redis.Redis(host = args["serverIP"], port = 6379)
widgets = [""]
pbar = [""]
for cam_id in camera_id:
    widgets.append(FormatLabel("| Camera "+cam_id+" FPS: %(value)d |"))
    pbar.append(ProgressBar(widgets=widgets))

while True:
    time.sleep(.1)
    for cam_id in camera_id:
        key = "fps_"+cam_id
        byte_get = BytesIO(store.get(key))
        int_get = np.load(byte_get)
        pbar[int(cam_id)].update(int_get)