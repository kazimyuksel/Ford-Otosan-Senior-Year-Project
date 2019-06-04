import cv2
assert cv2.__version__[0] == '3', 'The fisheye module requires opencv version >= 3.0.0'
import numpy as np
from time import sleep
video_capture = cv2.VideoCapture(0)
video_capture.set(3,640)
video_capture.set(4,480)
video_capture.set(5,20)




CHECKERBOARD = (9,6)
subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW
objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
_img_shape = None
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
K_master = np.zeros((3,3,1), dtype=np.float64)
D_master = np.zeros((4,1,1), dtype=np.float64)
counter = 1
file_dir = 'D:\\FordOtosan\\distorsion_calibration_images\\'
while True:
    ret, rgb = video_capture.read()
    #cv2.imshow('raw',rgb)
    gaussian_3 = cv2.GaussianBlur(rgb, (3,3), 10.0)
    rgb = cv2.addWeighted(rgb, 1.5, gaussian_3, -0.5, 0, rgb)
    #cv2.imshow('sharpen',rgb)
    rgb = cv2.bilateralFilter( rgb, 15, 120, 120 )
    #cv2.imshow('s&b',rgb)
    gaussian_3 = cv2.GaussianBlur(rgb, (3,3), 10.0)
    rgb = cv2.addWeighted(rgb, 1.5, gaussian_3, -0.5, 0, rgb)
    
    gray = cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY)
    img = rgb.copy()
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
    # If found, add object points, image points (after refining them)
    if ret == True:
        print('DONE '+str(counter))
        counter = counter+1
        objpoints.append(objp)
        cv2.cornerSubPix(gray,corners,(3,3),(-1,-1),subpix_criteria)
        imgpoints.append(corners)
    
    if  counter > 30 and counter % 5 == 0:
        objpoints = objpoints[4:]
        imgpoints = imgpoints[4:]
        N_OK = len(objpoints)
        K = np.zeros((3, 3))
        D = np.zeros((4, 1))
        rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
        tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
        rms, _, _, _, _ = \
            cv2.fisheye.calibrate(
                objpoints,
                imgpoints,
                gray.shape[::-1],
                K,
                D,
                rvecs,
                tvecs,
                calibration_flags,
                (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
            )
        #K_3d = K[np.newaxis,:,:]
        #D_3d = D[np.newaxis,:,:]
        #np.append(K_master, K_3d, axis =0)
        #np.append(D_master, D_3d, axis =0)
        K_master = np.dstack((K_master,K))
        D_master = np.dstack((D_master,D))
    #cv2.imshow('s&b&s',img)
    if cv2.waitKey(27) & 0xFF == ord('q') :
        K_master = K_master[:,:,1:]
        D_master = D_master[:,:,1:]
        D_mean = D_master.mean(axis=2) 
        K_mean = K_master.mean(axis=2)
        print("K=np.array(" + str(K_mean.tolist()) + ")")
        print("D=np.array(" + str(D_mean.tolist()) + ")")
        break

    sleep(0.1)
        
video_capture.release()
#cv2.destroyAllWindows()
