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

class PiVideoStream:
	def __init__(self, resolution=(640, 480), framerate=32):
		# initialize the camera and stream
		self.camera = PiCamera()
		self.camera.resolution = resolution
		self.camera.framerate = framerate
		self.rawCapture = PiRGBArray(self.camera, size=resolution)
		self.stream = self.camera.capture_continuous(self.rawCapture,
			format="bgr", use_video_port=True)
 
		# initialize the frame and the variable used to indicate
		# if the thread should be stopped
		self.frame = None
		self.stopped = False
    
	def start(self):
		# start the thread to read frames from the video stream
		Thread(target=self.update, args=()).start()
		return self
 
	def update(self):
		# keep looping infinitely until the thread is stopped
		for f in self.stream:
			# grab the frame from the stream and clear the stream in
			# preparation for the next frame
			self.frame = f.array
			self.rawCapture.truncate(0)
 
			# if the thread indicator variable is set, stop the thread
			# and resource camera resources
			if self.stopped:
				self.stream.close()
				self.rawCapture.close()
				self.camera.close()
				return
    
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
#CAMERA PARAMETERS

# Camera image dimension
DIM=(640, 480)
# Camera intrinsic matrix
K_1=np.array([[335.5619374393186, 0.0, 320.0], [0.0, 333.5403907261929, 240.0], [0.0, 0.0, 1.0]])
#Camera distorsion matrix
D_1=np.array([[-0.01877824020864759], [-0.036627875068484854], [0.03601968351612183], [-0.012209616189402786]])



###########################################
###########################################
###########################################
# MAIN PROGRAM STARTS HERE

CAMERA_FRAMERATE = 32
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
