import cv2
#assert cv2.__version__[0] == '3', 'The fisheye module requires opencv version >= 3.0.0'
import numpy as np
from time import sleep
DIM=(640, 480)
def undistort(img_array, balance=1.0, K=None, D=None, dim2=None, dim3=None):
    img = img_array
    dim1 = img.shape[:2][::-1]  #dim1 is the dimension of input image to un-distort
    assert dim1[0]/dim1[1] == DIM[0]/DIM[1], "Image to undistort needs to have same aspect ratio as the ones used in calibration"
    if not dim2:
        dim2 = dim1
    if not dim3:
        dim3 = dim1
    scaled_K = K * dim1[0] / DIM[0]  # The values of K is to scale with image dimension.
    scaled_K[2][2] = 1.0  # Except that K[2][2] is always 1.0
    # This is how scaled_K, dim2 and balance are used to determine the final K used to un-distort image. OpenCV document failed to make this clear!
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, D, dim2, np.eye(3), balance=balance)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, D, np.eye(3), new_K, dim3, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return undistorted_img

max_sleep = 5.0
cur_sleep = 0.1
while True:
    video_capture = cv2.VideoCapture("rtsp://admin:driveME2019@192.168.1.64/Streaming/Channels/2/picture")
    if video_capture.isOpened():
        break
    print('not opened, sleeping {}s'.format(cur_sleep))
    time.sleep(cur_sleep)
    if cur_sleep < max_sleep:
        cur_sleep *= 2
        cur_sleep = min(cur_sleep, max_sleep)
        continue
    cur_sleep = 0.1


file_dir = 'D:\\FordOtosan\\distorsion_calibration_images\\'


CHECKERBOARD = (9,6)
subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW
objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
_img_shape = None
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
K_master = np.zeros((3,3,0), dtype=np.float64)
D_master = np.zeros((4,1,0), dtype=np.float64)
counter = 1
file_dir = 'D:\\FordOtosan\\distorsion_calibration_images\\'
while True:
    ret, rgb = video_capture.read()
    if rgb is None:
        time.sleep(0.2)
        continue
    #cv2.imshow('raw',rgb)
    gaussian_3 = cv2.GaussianBlur(rgb, (3,3), 10.0)
    rgb = cv2.addWeighted(rgb, 1.5, gaussian_3, -0.5, 0, rgb)
    #cv2.imshow('sharpen',rgb)
    rgb = cv2.bilateralFilter( rgb, 15, 120, 120 )
    #cv2.imshow('s&b',rgb)
    gaussian_3 = cv2.GaussianBlur(rgb, (3,3), 10.0)
    rgb = cv2.addWeighted(rgb, 1.5, gaussian_3, -0.5, 0, rgb)
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
    
    if  counter > 9 and ret ==True and counter % 5 == 0:
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
                (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-9)
            )

        K_master = np.dstack((K_master,K))
        D_master = np.dstack((D_master,D))
        #D_mean = D_master.mean(axis=2) 
        #K_mean = K_master.mean(axis=2)
        img = undistort(rgb, K = K, D = D)
        path = file_dir + 'image_'+str(K_master.shape[2]-1)+'.jpg'
        cv2.imwrite(path,img)
        cv2.imshow('s&b&s',img)
    
    if cv2.waitKey(27) & 0xFF == ord('q') :
        #D_mean = D_master.mean(axis=2) 
        #K_mean = K_master.mean(axis=2)
        print("K=np.array(" + str(K_master[:,:,K_master.shape[2]-1].tolist()) + ")")
        print("D=np.array(" + str(D_master[:,:,D_master.shape[2]-1].tolist()) + ")")
        break
    sleep(0.5)
        
video_capture.release()
cv2.destroyAllWindows()
