import cv2
assert cv2.__version__[0] == '3', 'The fisheye module requires opencv version >= 3.0.0'
import numpy as np
from time import sleep
video_capture = cv2.VideoCapture(0)
video_capture.set(3,640)
video_capture.set(4,480)
video_capture.set(5,20)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-12)
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
    ret, corners = cv2.findChessboardCorners(gray, (9,6), cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
    img = rgb.copy()
    if ret == True:
        corners2 = cv2.cornerSubPix(gray,corners,(3,3),(-1,-1),criteria)
        # Draw and display the corners
        img = rgb.copy();
        img = cv2.drawChessboardCorners(img, (9,6), corners2,ret)
        path = file_dir + 'image_'+str(counter)+'.jpg'
        cv2.imwrite(path,rgb)
        print(counter)
        counter = counter + 1
        
    cv2.imshow('s&b&s',img)
    if cv2.waitKey(27) & 0xFF == ord('q') :
        break

    sleep(0.4)
        
video_capture.release()
cv2.destroyAllWindows()