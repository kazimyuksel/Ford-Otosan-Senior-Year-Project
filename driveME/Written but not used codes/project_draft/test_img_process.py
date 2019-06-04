from __future__ import division

import os
import sys
import math
import datetime
import time

import cv2
assert cv2.__version__[0] >= '3', 'The program requires OpenCV version greater than >= 3.0.0'

import numpy as np
from colorama import Fore, Style
from skimage.measure import block_reduce as pooling
from skimage import color,io,exposure,transform
from sklearn.cluster import KMeans
from progressbar import ProgressBar, Bar, AdaptiveETA, Percentage, SimpleProgress, Timer
import pickle
import uuid
from image.ImageProcess import ImageProcess

CAMERA_DIMENSION =[]
CAMERA_FPS =[]
CAMERA_DISTORSION_COEFFICIENTS = []
CAMERA_CAMERA_MATRIX = []

# [y[0] for y in parametersList].index('FPS')
parametersList = []
parametersList.append(['Universally Unique Identifier', uuid.uuid4().hex])
parametersList.append(['Date and Time of Calibration', datetime.datetime.now()])
parametersList.append(['Running Folder ID','D:\\FordOtosan\\'])
parametersList.append(['DIM', (640,480)])
parametersList.append(['FPS', 20       ])
parametersList.append(['External Cameras Indices', None])
parametersList.append(['Camera Matrix', np.array([[433.1789201825628, 0.0, 336.0326551869997], [0.0, 430.52735997037956, 216.5706380556904], [0.0, 0.0, 1.0]])])
parametersList.append(['Distorsion Coefficients', np.array([[-0.05280848062554133], [-0.13174782579340452], [-0.028567480262252885], [0.060944836127339065]])])
parametersList.append(['ARUCO Calibration ID', 12])
parametersList.append(['ARUCO Head ID', 20])
parametersList.append(['ARUCO Trailer ID', 30])
parametersList.append(['ARUCO GROUND LEVEL CALIBRATION CORNERS', None])
parametersList.append(['ARUCO TOP LEVEL CALIBRATION CORNERS', None])

with open('D:\FordOtosan\\PARAMETERS.driveME', 'wb') as fp:
    pickle.dump(parametersList, fp)
    
with open ('D:\\FordOtosan\\PARAMETERS1.driveME', 'rb') as fp:
    itemlist = pickle.load(fp)
    
image = ImageProcess()
a =image.connectedCameraCount()
b =image.detectExternalCamera()
c =image.initializeExternalCamera()
#image.setCameraDIM()
#image.setCameraFPS()
#image.setCameraMatrix()
#image.setDistorsionCoefficients()
d =image.readExternalCamera()
e =image.releaseExternalCamera()