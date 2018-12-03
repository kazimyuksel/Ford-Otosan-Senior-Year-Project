import smbus
import time
import math
import cv2
assert cv2.__version__[0] == '3', 'The program requires OpenCV version greater than >= 3.0.0'
import numpy as np
import os
from threading import Thread
from picamera.array import PiRGBArray
from picamera import PiCamera
import imutils
from imutils.video.pivideostream import PiVideoStream
from imutils.video import FPS
import argparse
    
###########################################
# MPU6050 RELATED REGISTERS AND FUCNTIONS

# Power management registers
power_mgmt_1 = 0x6b
power_mgmt_2 = 0x6c

# Read single byte from i2c BUS
def read_byte(adr):
    return bus.read_byte_data(address, adr)

# Read double byte from i2c BUS
def read_word(adr):
    high = bus.read_byte_data(address, adr)
    low = bus.read_byte_data(address, adr+1)
    val = (high << 8) + low
    return val

def read_word_2c(adr):
    val = read_word(adr)
    if (val >= 0x8000):
        return -((65535 - val) + 1)
    else:
        return val

# Distance Calculation
def dist(a,b):
    return math.sqrt((a*a)+(b*b))

# Angle of Y axis, pitch
def get_y_rotation(x,y,z):
    radians = math.atan2(x, dist(y,z))
    return -math.degrees(radians)

# Angle of X axis, roll
def get_x_rotation(x,y,z):
    radians = math.atan2(y, dist(x,z))
    return math.degrees(radians)

###########################################
#CAMERA PARAMETERS

# Camera image dimension
DIM=(640, 480)
# Camera intrinsic matrix
K_1=np.array([[335.5619374393186, 0.0, 320.0], [0.0, 333.5403907261929, 240.0], [0.0, 0.0, 1.0]])
#Camera distorsion matrix
D_1=np.array([[-0.01877824020864759], [-0.036627875068484854], [0.03601968351612183], [-0.012209616189402786]])

###########################################
#CAMERA FUNCTIONS

# Undistort Image
def undistort(img_array, balance=1.0, dim2=None, dim3=None):
    img = img_array
    dim1 = img.shape[:2][::-1]  #dim1 is the dimension of input image to un-distort
    assert dim1[0]/dim1[1] == DIM[0]/DIM[1], "Image to undistort needs to have same aspect ratio as the ones used in calibration"
    if not dim2:
        dim2 = dim1
    if not dim3:
        dim3 = dim1
    scaled_K = K_1 * dim1[0] / DIM[0]  # The values of K is to scale with image dimension.
    scaled_K[2][2] = 1.0  # Except that K[2][2] is always 1.0
    # This is how scaled_K, dim2 and balance are used to determine the final K used to un-distort image. OpenCV document failed to make this clear!
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, D_1, dim2, np.eye(3), balance=balance)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, D_1, np.eye(3), new_K, dim3, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return undistorted_img


###########################################
###########################################
###########################################
# MAIN PROGRAM STARTS HERE

CAMERA_FRAMERATE = 90
CAMERA_HEIGHT_PIXEL = 480
CAMERA_WIDTH_PIXEL = 640
camera_H_angle = 82 #degree
camera_W_angle = 113 #degree
compression_factor_H = 1
compression_factor_W = 1
camera_height = 100 
top_crop_pixel = 50
bottom_crop_pixel = 0
left_crop_pixel = 0
right_crop_pixel = 0

# Initialize i2c connection
bus = smbus.SMBus(1)
address = 0x68 #MPU6050 i2c address

# MPU6050 wakeup command
bus.write_byte_data(address, power_mgmt_1, 0)


stream = PiVideoStream(resolution=(CAMERA_WIDTH_PIXEL, CAMERA_HEIGHT_PIXEL), CAMERA_FRAMERATE=32).start()
# Allow the camera to warmup
time.sleep(2.0)

start_time = 0
avg_hz = 0

H = CAMERA_HEIGHT_PIXEL - top_crop_pixel - bottom_crop_pixel
W = CAMERA_WIDTH_PIXEL - left_crop_pixel - right_crop_pixel

# Get raw accelaration data from MPU6050
accel_xout = read_word_2c(0x3b)
accel_yout = read_word_2c(0x3d)
accel_zout = read_word_2c(0x3f)

# Convert accelaration data to m/s^2
accel_xout_scaled = accel_xout / 16384.0
accel_yout_scaled = accel_yout / 16384.0
accel_zout_scaled = accel_zout / 16384.0

# Get initial camera angle
camera_theta = 90 + get_y_rotation(accel_xout_scaled, accel_yout_scaled, accel_zout_scaled)

while True:
    accel_xout = read_word_2c(0x3b)
    accel_yout = read_word_2c(0x3d)
    accel_zout = read_word_2c(0x3f)
    
    accel_xout_scaled = accel_xout / 16384.0
    accel_yout_scaled = accel_yout / 16384.0
    accel_zout_scaled = accel_zout / 16384.0
    
    # Low Pass Filter
    lpf_camera = 0.6
    camera_theta = camera_theta*lpf_camera + (90 + get_y_rotation(accel_xout_scaled, accel_yout_scaled, accel_zout_scaled))*(1-lpf_camera)
    
    # Calculate side length of square to be transformed
    L_ref = 2*camera_height*np.tan(np.radians(camera_H_angle)/5)
    
    # Calculate pixel per radians
    h = H/np.radians(camera_H_angle)
    w = W/np.radians(camera_W_angle)
    
    # Distance to center of camera
    x_mid = camera_height/np.tan(np.radians(camera_theta))
    
    # Lateral Line's angle to midpoint
    phi_1 = np.arccos((camera_height*camera_height + x_mid*(x_mid-L_ref/2))/((np.sqrt(np.square(camera_height)+np.square(x_mid)))*(np.sqrt(np.square(camera_height)+np.square(x_mid-L_ref/2)))))
    phi_2 = np.arccos((camera_height*camera_height + x_mid*(x_mid+L_ref/2))/((np.sqrt(np.square(camera_height)+np.square(x_mid)))*(np.sqrt(np.square(camera_height)+np.square(x_mid+L_ref/2)))))
    
    # Longitudinal Line's angle to midpoint
    beta_1 = np.arctan((L_ref/2)/(np.sqrt(np.square(x_mid-L_ref/2)+np.square(camera_height))))
    beta_2 = np.arctan((L_ref/2)/(np.sqrt(np.square(x_mid+L_ref/2)+np.square(camera_height))))
    
    # Corresponing corner's pixel values
    I_t = H/2 - compression_factor_H*phi_1*h
    I_b = H/2 + compression_factor_H*phi_2*h
    J_tr = W/2 + compression_factor_W*beta_2*w
    J_tl = W/2 - compression_factor_W*beta_2*w
    J_br = W/2 + compression_factor_W*beta_1*w
    J_bl = W/2 - compression_factor_W*beta_1*w
    
    # Array of corner coordinates for line drawing
    ### This part will not be useful in future
    p_1_prime = (np.round(J_tr).astype(int),np.round(I_t).astype(int))
    p_2_prime = (np.round(J_br).astype(int),np.round(I_b).astype(int))
    p_3_prime = (np.round(J_bl).astype(int),np.round(I_b).astype(int))
    p_4_prime = (np.round(J_tl).astype(int),np.round(I_t).astype(int))
    
    # Array of corner coordinates and transformed square corner coordinates
    pts1 = np.float32([[J_tr,I_t],[J_br,I_b],[J_bl,I_b],[J_tl,I_t]])
    pts2 = np.float32([[540,470],[540,530],[460,530],[460,470]])
    
    # Homology Matrix for transformation
    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    image = vs.read()

    image = undistort(image)
    
    # Crop image to given spesifications
    image_2 = image[top_crop_pixel:(CAMERA_HEIGHT_PIXEL-bottom_crop_pixel),left_crop_pixel:(CAMERA_WIDTH_PIXEL-right_crop_pixel)]
    
    # Draw perspective square on image
    ### Will be removed in future
    cv2.line(image, p_1_prime, p_2_prime, (0,255,0), thickness=2, lineType=8, shift=0)
    cv2.line(image, p_2_prime, p_3_prime, (0,255,0), thickness=2, lineType=8, shift=0)
    cv2.line(image, p_3_prime, p_4_prime, (0,255,0), thickness=2, lineType=8, shift=0)
    cv2.line(image, p_4_prime, p_1_prime, (0,255,0), thickness=2, lineType=8, shift=0)

    # Show undistorted image with lines
    cv2.imshow("Undistort", image)
    
    # Apply Inverse Perspective Transform
    image_3 = cv2.warpPerspective(image_2, matrix, (1000,1000))
    
    #Show Inverse Perspective Transformed image
    cv2.imshow("Advanced Undistort", image_3)
	
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    
    # Calculate Looptime and Frequency
    elapsed_time = time.time() - start_time
    start_time = time.time()  
    avg_hz = avg_hz*0.8 + 0.2/elapsed_time
    print(avg_hz)
	
	
cv2.destroyAllWindows()
vs.stop()
	

