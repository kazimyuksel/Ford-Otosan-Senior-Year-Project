import os
import cv2
from cv2 import aruco
import numpy as np

def draw(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)

    # draw ground floor in green
    img = cv2.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)

    # draw pillars in blue color
    for i,j in zip(range(4),range(4,8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)

    # draw top layer in red color
    img = cv2.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)

    return img
axis = np.float32([[-3,-3,0], [3,-3,0], [3,3,0], [-3,3,0], [-3,-3,6], [3,-3,6], [3,3,6], [-3,3,6]])*5/2
file_dir = 'D:\\FordOtosan\\distorsion_calibration_images\\'
video_capture = cv2.VideoCapture(0)
video_capture.set(3,640)
video_capture.set(4,480)
video_capture.set(5,20)


DIM = (640,480)
K =np.array([[433.1789201825628, 0.0, 336.0326551869997], [0.0, 430.52735997037956, 216.5706380556904], [0.0, 0.0, 1.0]])
D=np.array([[-0.05280848062554133], [-0.13174782579340452], [-0.028567480262252885], [0.060944836127339065]])

dim1 = DIM
dim2 = (640, 480)
dim3 = (640, 480)
balance=1.0

bool_start_1 = True
bool_start_2 = True

_width  = 30.0
_height = 30.0
_margin = 470.0

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

aruco_dict = aruco.Dictionary_get( aruco.DICT_6X6_1000 )
markerLength = 15.0
arucoParams = aruco.DetectorParameters_create()



criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-12)
counter = 1
file_dir = 'D:\\FordOtosan\\distorsion_calibration_images\\'
while True:
    ret, rgb = video_capture.read()
    rgb = cv2.remap(rgb, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    imgRemapped_gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    corners, ids, rejectedImgPoints = aruco.detectMarkers(imgRemapped_gray, aruco_dict, parameters=arucoParams)
    if ids != None: # if aruco marker detected
        a = ids
        imgWithAruco = rgb
        
    else:   # if aruco marker is NOT detected
        imgWithAruco = rgb  # assign imRemapped_color to imgWithAruco directly

    if cv2.waitKey(27) & 0xFF == ord('q') :
        break
    cv2.imshow("aruco", imgWithAruco)
    
video_capture.release()
cv2.destroyAllWindows()


