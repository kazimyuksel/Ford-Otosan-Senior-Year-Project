import os
import cv2
from cv2 import aruco
import numpy as np
import time

calibration_aruco_id = 15
calib_marker_length = 15
head_aruco_id = 27
head_marker_length = 15

global_head_points = np.array([], dtype=np.int32).reshape(0,2)

bool_draw_marker = False

file_dir = 'D:\\FordOtosan\\distorsion_calibration_images\\'
video_capture = cv2.VideoCapture(0)
video_capture.set(3,640)
video_capture.set(4,480)
video_capture.set(5,20)

video_capture_2 = cv2.VideoCapture(2)
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

_width  = 15.0
_height = 15.0
_margin = 235.0

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
start_time = time.time()
avg_hz = 0
while True:
    ret, rgb = video_capture.read()
    ret_2, rgb_2 = video_capture_2.read()
    
    rgb = cv2.remap(rgb, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    rgb_2 = cv2.remap(rgb_2, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    
    imgRemapped_gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    imgRemapped_gray_2 = cv2.cvtColor(rgb_2, cv2.COLOR_BGR2GRAY)
    
    
    corners, ids, rejectedImgPoints = aruco.detectMarkers(imgRemapped_gray, aruco_dict, parameters=arucoParams)
    corners_2, ids_2, rejectedImgPoints_2 = aruco.detectMarkers(imgRemapped_gray_2, aruco_dict, parameters=arucoParams)
    
    if ids is not None and bool_draw_marker == True:
        imgWithAruco = aruco.drawDetectedMarkers(rgb, corners, ids, (0,255,0))
    else:
        imgWithAruco = rgb
    if ids_2 is not None and bool_draw_marker == True:
        imgWithAruco_2 = aruco.drawDetectedMarkers(rgb_2, corners_2, ids_2, (0,255,0))
    else:
        imgWithAruco_2 = rgb_2
    
    
    
    if ids is None:
        calibration_index = []
        calibration_index = np.array(calibration_index)
        head_index = []
        head_index = np.array(head_index).astype(np.int32)
        #calibration_index.size
    else:
        ids_list = np.ndarray.tolist(ids)
        ids_array = np.array(ids_list)
        ids_array = ids_array.T
        ids_array = ids_array[0,:]
        calibration_index = np.where(ids_array == 15)
        head_index = np.where(ids_array == 27)
        calibration_index = calibration_index[0]
        head_index = head_index[0]
        
    
    if ids_2 is None:
        calibration_index_2 = []
        calibration_index_2 = np.array(calibration_index_2).astype(np.int32)
        head_index_2 = []
        head_index_2 = np.array(head_index_2).astype(np.int32)
        #calibration_index_2.size
    else:
        ids_list_2 = np.ndarray.tolist(ids_2)
        ids_array_2 = np.array(ids_list_2)
        ids_array_2 = ids_array_2.T
        ids_array_2 = ids_array_2[0,:]
        calibration_index_2 = np.where(ids_array_2 == 15)
        head_index_2 = np.where(ids_array_2 == 27)
        calibration_index_2 = calibration_index_2[0]
        head_index_2 = head_index_2[0]
        
    
    if calibration_index.size == 1: # if aruco marker detected
        calib_index = np.asscalar(calibration_index)
        calib_corner = corners[calib_index]
        pts_src = np.array( calib_corner, np.float32 )
        #pts_src = pts_src[0,:,:,:]
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
        
    if calibration_index_2.size == 1: # if aruco marker detected
        calib_index_2 = np.asscalar(calibration_index_2)
        #calib_id_2 = ids_2[calibration_index_2].reshape((1,1))
        calib_corner_2 = corners_2[calib_index_2]
        #rvec_2, tvec_2 , pose_2 = aruco.estimatePoseSingleMarkers(calib_corner_2, calib_marker_length, K, D) # For a single marker
        #rvec_2 = rvec_2[0,:,:]
        #tvec_2 = tvec_2[0,:,:]
        #imgWithAruco_2 = aruco.drawDetectedMarkers(rgb_2, calib_corner_2, calib_id_2, (0,255,0))
        #imgWithAruco = aruco.drawAxis(imgWithAruco, K, D, rvec, tvec, 20)    # axis length 100 can be changed according to your requirement
           # display
        pts_src_2 = np.array( calib_corner_2, np.float32 )
        #pts_src_2 = pts_src_2[0,:,:,:]
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
    
    
    if (calibration_index.size == 1) and (calibration_index_2.size == 1):
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
        
        
    if (head_index.size == 1) and (head_index_2.size == 1) and (calibration_index.size == 1) and (calibration_index_2.size == 1):
        head_scalar_index = np.asscalar(head_index)
        head_scalar_index_2 = np.asscalar(head_index_2)
        
        head_corner = corners[head_scalar_index]
        head_corner_2 = corners_2[head_scalar_index_2]
        
        pts_head = np.array( head_corner, np.float32 )
        pts_head_2 = np.array( head_corner_2, np.float32 )
        
        pts_head_mean = np.array([pts_head.mean(axis=1)])
        pts_head_mean_2 = np.array([pts_head_2.mean(axis=1)])
        
        global_head_mean_point = cv2.perspectiveTransform(pts_head_mean,h)
        global_head_mean_point_2 = cv2.perspectiveTransform(pts_head_mean_2,h_2)
        global_head_point = np.mean([global_head_mean_point,global_head_mean_point_2],axis = 0).astype(np.int32)
        global_head_point = global_head_point[0,:,:]
        global_head_points = np.vstack([global_head_points, global_head_point])
        
    elif (head_index.size == 1) and (head_index_2.size != 1) and (calibration_index.size == 1) and (calibration_index_2.size == 1):
        head_scalar_index = np.asscalar(head_index)
        head_corner = corners[head_scalar_index]
        pts_head = np.array( head_corner, np.float32 )
        pts_head_mean = np.array([pts_head.mean(axis=1)])
        global_head_point = cv2.perspectiveTransform(pts_head_mean,h)
        global_head_point = global_head_point[0,:,:].astype(np.int32)
        global_head_points = np.vstack([global_head_points, global_head_point])
        
    elif (head_index.size != 1) and (head_index_2.size == 1) and (calibration_index.size == 1) and (calibration_index_2.size == 1):
        head_scalar_index_2 = np.asscalar(head_index_2)
        head_corner_2 = corners_2[head_scalar_index_2]
        pts_head_2 = np.array( head_corner_2, np.float32 )
        pts_head_mean_2 = np.array([pts_head_2.mean(axis=1)])
        global_head_point = cv2.perspectiveTransform(pts_head_mean_2,h_2)
        global_head_point = global_head_point[0,:,:].astype(np.int32)
        global_head_points = np.vstack([global_head_points, global_head_point])

        
    if (calibration_index.size == 1) and (calibration_index_2.size == 1):
        final_img = cv2.polylines(final_img,[global_head_points],False,(0,255,0),2)
        cv2.imshow("stitched", final_img)
        
    
    cv2.imshow("orj", rgb)
    cv2.imshow("orj_2", rgb_2)
    elapsed_time = time.time() - start_time
    start_time =time.time()
    avg_hz = 0.8*avg_hz + 0.2/elapsed_time
    print(avg_hz)
    if cv2.waitKey(27) & 0xFF == ord('q') :
        break
       
video_capture.release()
video_capture_2.release()
cv2.destroyAllWindows()


