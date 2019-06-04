
from __future__ import division


import time

import cv2
assert cv2.__version__[0] >= '3', 'The program requires OpenCV version greater than >= 3.0.0'
from cv2 import aruco
import numpy as np
from io import BytesIO
import redis

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

_width  = 35.0
_height = 35.0
_margin = 300.0

pts_dst_corner = np.array(
	[
		[[  		_margin, _margin 			]],
		[[ 			_margin, _height + _margin  ]],
		[[ _width + _margin, _height + _margin  ]],
		[[ _width + _margin, _margin 			]],
	]
)
pts_dst = np.array( pts_dst_corner, np.float32 )
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

    if ids is not None:
        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, markerLength, scaled_K, D)
        imgWithAruco = aruco.drawDetectedMarkers(rgb, corners, ids, (0,255,0))
        pts_src = np.array( corners, np.float32 )
        pts_src = pts_src[0,:,:,:]
        pts_src = np.rollaxis(pts_src,1)
        h, status = cv2.findHomography( pts_src, pts_dst )
        corners_array = np.concatenate([corners_array,corners[0]],axis = 0)
        counter = counter +1
        out = cv2.warpPerspective( rgb, h, ( int( _width + _margin * 2 ), int( _height + _margin * 2 ) ) )
        cv2.imshow('bird', out)
    else:
        imgWithAruco = rgb
    cv2.imshow('occupancy', img_smooth)
    
    elapsed_time = time.time() - start_time
    start_time =time.time()
    avg_hz = 0.8*avg_hz + 0.2/elapsed_time
    print(avg_hz)
    if counter == 100:
        mean_corner = corners_array.mean(axis=0)
        break
    if cv2.waitKey(27) & 0xFF == ord('q') :
        break
    
cv2.destroyAllWindows()



