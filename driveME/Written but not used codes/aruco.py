import os
import cv2
from cv2 import aruco
import numpy as np
file_dir = 'D:\\FordOtosan\\distorsion_calibration_images\\'
video_capture = cv2.VideoCapture(0)
video_capture.set(3,640)
video_capture.set(4,480)
video_capture.set(5,20)

video_capture_2 = cv2.VideoCapture(1)
video_capture_2.set(3,640)
video_capture_2.set(4,480)
video_capture_2.set(5,20)
DIM = (640,480)
K=np.array([[433.1789201825628, 0.0, 336.0326551869997], [0.0, 430.52735997037956, 216.5706380556904], [0.0, 0.0, 1.0]])
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
    ret_2, rgb_2 = video_capture_2.read()
    rgb = cv2.remap(rgb, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    rgb_2 = cv2.remap(rgb_2, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    imgRemapped_gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    imgRemapped_gray_2 = cv2.cvtColor(rgb_2, cv2.COLOR_BGR2GRAY)
    corners, ids, rejectedImgPoints = aruco.detectMarkers(imgRemapped_gray, aruco_dict, parameters=arucoParams)
    corners_2, ids_2, rejectedImgPoints_2 = aruco.detectMarkers(imgRemapped_gray_2, aruco_dict, parameters=arucoParams)
    if ids != None: # if aruco marker detected
        rvec, tvec , pose = aruco.estimatePoseSingleMarkers(corners, markerLength, K, D) # For a single marker
        rvec = rvec[0,:,:]
        tvec = tvec[0,:,:]
        imgWithAruco = aruco.drawDetectedMarkers(rgb, corners, ids, (0,255,0))
        #imgWithAruco = aruco.drawAxis(imgWithAruco, K, D, rvec, tvec, 20)    # axis length 100 can be changed according to your requirement
           # display
        
            
        pts_src = np.array( corners, np.float32 )
        pts_src = pts_src[0,:,:,:]
        pts_src = np.rollaxis(pts_src,1)
        if bool_start_1 == True:
            temp = pts_src
            bool_start_1 = False
        temp = temp*0.9 + pts_src*0.1
        h, status = cv2.findHomography( temp, pts_dst )
        out = cv2.warpPerspective( rgb, h, ( int( _width + _margin * 2 ), int( _height + _margin * 2 ) ) )
        gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        _,gray_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
        
        cv2.imshow("birds_view_1", out)
    else:   # if aruco marker is NOT detected
        imgWithAruco = rgb  # assign imRemapped_color to imgWithAruco directly
        
    if ids_2 != None: # if aruco marker detected
        rvec_2, tvec_2 , pose_2 = aruco.estimatePoseSingleMarkers(corners_2, markerLength, K, D) # For a single marker
        rvec_2 = rvec_2[0,:,:]
        tvec_2 = tvec_2[0,:,:]
        imgWithAruco_2 = aruco.drawDetectedMarkers(rgb_2, corners_2, ids_2, (0,255,0))
        #imgWithAruco = aruco.drawAxis(imgWithAruco, K, D, rvec, tvec, 20)    # axis length 100 can be changed according to your requirement
           # display
        pts_src_2 = np.array( corners_2, np.float32 )
        pts_src_2 = pts_src_2[0,:,:,:]
        pts_src_2 = np.rollaxis(pts_src_2,1)
        if bool_start_2 == True:
            temp_2 = pts_src_2
            bool_start_2 = False
        temp_2 = temp_2*0.9 + pts_src_2*0.1
        h_2, status_2 = cv2.findHomography( temp_2, pts_dst )
        out_2 = cv2.warpPerspective( rgb_2, h_2, ( int( _width + _margin * 2 ), int( _height + _margin * 2 ) ) )
        gray_2 = cv2.cvtColor(out_2, cv2.COLOR_BGR2GRAY)
        _,gray_thresh_2 = cv2.threshold(gray_2, 0, 255, cv2.THRESH_BINARY)
        cv2.imshow("birds_view_2", out_2)
    else:   # if aruco marker is NOT detected
        imgWithAruco = rgb  # assign imRemapped_color to imgWithAruco directly
    
    if ids_2 != None and ids != None:
        xor_mask = cv2.bitwise_xor(gray_thresh_2,gray_thresh)
        and_mask = cv2.bitwise_and(gray_thresh_2,gray_thresh)
        #and_mask = cv2.GaussianBlur(and_mask,(3, 3), 0)
        #xor_mask = cv2.GaussianBlur(xor_mask,(3, 3), 0)
        added_raw_image = cv2.addWeighted(out,0.5,out_2,0.5, gamma = 0.0)
        masked_1 = cv2.bitwise_and(out, out, mask=xor_mask)
        masked_2 = cv2.bitwise_and(out_2, out_2, mask=xor_mask)
        xor_img = cv2.add(masked_1,masked_2)
        and_img = cv2.bitwise_and(added_raw_image, added_raw_image, mask=and_mask)
        final_img = cv2.add(xor_img,and_img)
        cv2.imshow("stitched", final_img)
    
    
    cv2.imshow("orj", rgb)
    cv2.imshow("orj_2", rgb_2)
    
    if cv2.waitKey(27) & 0xFF == ord('q') :
        break
    
    
video_capture.release()
video_capture_2.release()
cv2.destroyAllWindows()


