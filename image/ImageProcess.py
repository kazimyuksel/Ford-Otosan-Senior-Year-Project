# -*- coding: utf-8 -*-

"""
This module has required functions(modules) for image processing under the class called "ImageProcess". It will be used
in parts where inverse perspective transform, object tracking, image stitching, occupancy grid calculations required.

This class is only for manipulating existing IMAGE ARRAY, it does not take any array from the buffer of the running
system.
"""

# Author: Kazım Yüksel <kazim.yuksel95@gmail.com>
# It is protected under GNU General Public License v3.0

# This Module is coded for Bilkent University Mechanical Engineering Senior Design Project 2019, Group B3
# Industrial advisor, sponsor and project idea owner is Ford Otosan <https://www.fordotosan.com.tr/>
# Intellectual property of this "ImageProcess.py" module belongs to DriveMe group <driveme@googlegroups.com>
# Following code can be used, modified freely. Auther must notified with any improvements.
# Code can be used with proper cititation of auther and github repository link in order to increase accesibility
# and possible debuging.

from __future__ import division

import os
import sys
import math
import datetime
import time

import cv2
assert cv2.__version__[0] == '3', 'The program requires OpenCV version greater than >= 3.0.0'

import numpy as np
from colorama import Fore, Style
from skimage.measure import block_reduce as pooling
from skimage import color,io,exposure,transform
from sklearn.cluster import KMeans
from progressbar import ProgressBar, Bar, AdaptiveETA, Percentage, SimpleProgress, Timer

class ImageProcess():
    """
    Initialization Module to be used inside class: __init__ : Initializes ImageProcess class with given constants which
                                                              can be changed in future depending on the need.

    Public Modules to be used inside class: crop
                                            pool
                                            weightedAdd
                                            gaussianBlur
                                            canny
                                            sobel
                                            connectedCameraCount
                                            detectExternalCamera
                                            initializeExternalCamera
                                            releaseExternalCamera
                                            readExternalCamera
                                            setCameraMatrix
                                            getCameraMatrix
                                            setDistorsionCoefficients
                                            getDistorsionCoefficients
                                            setCameraDIM
                                            getCameraDIM
                                            setCameraFPS
                                            getCameraFPS                                           
                                            calibrateParameters
                                            readParameters
                                            setParameters




    Private Modules to be used inside class: _bgr2gray      : Converts Blue,Green,Red to Grayscale
                                             _bgr2rgb       : Converts Blue,Green,Red to Red,Green,Blue
                                             _bgr2hsv       : Converts Blue,Green,Red to Hue,Saturation,Value
                                             _bgr2lab       : Converts Blue,Green,Red to Lightness,a,b
                                             _bgr2YCrCb     : Converts Blue,Green,Red to Luminance,Cr,Cb

                                             _rgb2gray      : Converts Red,Green,Blue to Grayscale
                                             _rgb2bgr       : Converts Red,Green,Blue to Blue,Green,Red
                                             _rgb2hsv       : Converts Red,Green,Blue to Hue,Saturation,Value
                                             _rgb2lab       : Converts Red,Green,Blue to Lightness,a,b
                                             _rgb2YCrCb     : Converts Red,Green,Blue to Luminance,Cr,Cb

                                             _hsv2gray      : Converts Hue,Saturation,Value to Grayscale
                                             _hsv2bgr       : Converts Hue,Saturation,Value to Blue,Green,Red
                                             _hsv2rgb       : Converts Hue,Saturation,Value to Red,Green,Blue
                                             _hsv2lab       : Converts Hue,Saturation,Value to Lightness,a,b
                                             _hsv2YCrCb     : Converts Hue,Saturation,Value to Luminance,Cr,Cb

                                             _lab2gray      : Converts Lightness,a,b to Grayscale
                                             _lab2bgr       : Converts Lightness,a,b to Blue,Green,Red
                                             _lab2hsv       : Converts Lightness,a,b to Hue,Saturation,Value
                                             _lab2rgb       : Converts Lightness,a,b to Red,Green,Blue
                                             _lab2YCrCb     : Converts Lightness,a,b to Luminance,Cr,Cb

                                             _YCrCb2gray    : Converts Luminance,Cr,Cb to Grayscale
                                             _YCrCb2bgr     : Converts Luminance,Cr,Cb to Blue,Green,Red
                                             _YCrCb2hsv     : Converts Luminance,Cr,Cb to Hue,Saturation,Value
                                             _YCrCb2lab     : Converts Luminance,Cr,Cb to Lightness,a,b
                                             _YCrCb2rgb     : Converts Luminance,Cr,Cb to Red,Green,Blue

                                             _histogramNormalize : Normalize Value channel of HSV. It take input image
                                                                   format as string with default value as HSV. Formats
                                                                   can be following; 'hsv'
                                                                                     'bgr'
                                                                                     'rgb'
                                                                                     'lab'
                                                                                     'ycrcb'
                                             __version__ : Print version of module
                                             __author__  : Print author of module
                                             __cv2version__ : Print openCV version

    """

    #Initialize class with either DEFAULT variables or given.
    def __init__(self,
                 gaussianBlur_kernelSize=5,
                 poolDimension=2,
                 crop_topPixel=0,
                 crop_bottomPixel=0,
                 crop_leftPixel=0,
                 crop_rightPixel=0):

        self._version ='0.1'
        self._author ='Kazım Yüksel <kazim.yuksel95@gmail.com>'
        self._cv2version = cv2.__version__
        
        self.projectFile = 'D:\FordOtosan'
        self.isFileExist = False
        self.isFileExist = os.path.exists(self.projectFile)
        if self.isFileExist is False:
            raise ValueError('A file with name: '+self.projectFile+' should exist.')
        
        self.gaussianBlur_kernelSize = gaussianBlur_kernelSize
        self.poolDimension = poolDimension
        self.crop_topPixel = crop_topPixel
        self.crop_bottomPixel = crop_bottomPixel
        self.crop_leftPixel = crop_leftPixel
        self.crop_rightPixel = crop_rightPixel
        
        self.CameraMatrix = None
        self.DistorsionCoefficients = None
        self.CameraDIM = (640,480)#None
        self.CameraFPS = None
        self.CAMERA_INDICES = None
        self._NUMBER_OF_CONNECTED_CAMERAS = None
        self._NUMBER_OF_FRAMES_FOR_EXTERNAL_CAMERA_DETECTION = 25
        self._EXTERNAL_CAMERA_INDEX_LIST_ = []
        self._ELEMINATION_THRESHOLD_ =  65
        
        self.CAMERA_LIST = []
        self.IMAGE_LIST = []
        
        self.isCameraDetected = False
        self.isExternalCameraDetecred = False
        self.isCameraInitialized = False
        self.isCameraOpen = False
        self.isCameraReleased = False
        self.isCameraMatrixSet = False
        self.isDistorsionCoefficientsSet = False
        
    def crop(self,img_arr,
             crop_topPixel=None,
             crop_bottomPixel=None,
             crop_leftPixel=None,
             crop_rightPixel=None):

        if (crop_topPixel == None):
            crop_topPixel = self.crop_topPixel
        if (crop_bottomPixel == None):
            crop_bottomPixel = self.crop_bottomPixel
        if (crop_leftPixel == None):
            crop_leftPixel = self.crop_leftPixel
        if (crop_rightPixel == None):
            crop_rightPixel = self.crop_rightPixel

        self.crop_ndim = img_arr.ndim
        if (self.crop_ndim == 2):
            height,width = img_arr.shape
            img_arr = img_arr[self.crop_topPixel:height-self.crop_bottomPixel,
                              self.crop_leftPixel:width-self.crop_rightPixel]
        elif (self.crop_ndim == 3):
            height,width,depth = img_arr.shape
            img_arr = img_arr[self.crop_topPixel:height-self.crop_bottomPixel,
                              self.crop_leftPixel:width-self.crop_rightPixel,
                              0:depth]
        else:
            print(self.crop_ndim)
            raise ValueError('Input is not a image, it is either 1D array or 4D tensor... I hope it is not a tensor.')

    #Pooling pixels of the given image.
    #After pooling with (3,3) kernel size, that 9 pixel is reduced to 1 pixel with maximum value.
    def pool(self,img_arr,poolDim=None):
        if (poolDim == None):
            poolDim = self.poolDimension
        return pooling(img_arr, (poolDim,poolDim), np.max)

    #Thresholding depending on the need.
    def thresholding(self,img_arr,type='BINARY'):

        self.thresholdingType=type
        return img_arr

    def weightedAdd(self,img_arr_1,img_arr_2,ratio=0.5,gamma=0):
        """
        Take 2 IMAGE array to be added with percentage given. Percentage is the FIRST input's ratio.
        :param img_arr_1: Input as NDIM NUMPY array.
        :param img_arr_2: Input as NDIM NUMPY array.
        :param ratio: Ratio of first image to be added. (1-ratio) is the second image's ratio.
        :param gamma: Added white to image. Generally 0.05 but initialized as zero. Can be changed.
        :return: Add 2 image on top with top layer opacity definded with "ratio". Output is NDIM as input.
        """
        self.weightedAddRatio = ratio
        self.weightedAddGamma = gamma
        if (img_arr_1.shape == img_arr_2.shape):
            if (self.weightedAddRatio <= 1 and self.weightedAddRatio >= 0):
                return cv2.addWeighted(img_arr_1, self.weightedAddRatio, img_arr_2,
                                       (1-self.weightedAddRatio), self.weightedAddGamma)
            else:
                raise ValueError('Given RATIO is not between 0-1, or 0-100 percent.')
        else:
            raise ValueError('Given ARRAY\'s NDIM does not match. Check uses of this module.')


    def gaussianBlur(self,img_arr,kernelSize = None):
        """
        Add GAUSSIAN BLUR to Image.
        :param img_arr: Input NDIM NUMPY array.
        :param gb_kernelSize: INTEGER for kernel size. It is initialized in __init__ but also can be changed each time.
        :return: returns NDIM NUMPY array.
        """
        if (kernelSize==None):
            kernelSize = self.gaussianBlur_kernelSize
        return cv2.GaussianBlur(img_arr,(kernelSize, kernelSize), 0)


    def canny(self,img_arr):
        """
        Canny Edge Detector with OTSU Adaptive Thresholding
        :param img_arr: Input NDIM NUMPY array.
        :return: returns NDIM NUMPY array in Black-White (0,255)
        """
        self.otsu, _ = cv2.threshold(img_arr, 0, 255, cv2.THRESH_BINARY or cv2.THRESH_OTSU)
        self.low_threshold = self.otsu * 0.5
        self.high_threshold = self.otsu
        return cv2.Canny(img_arr,self.low_threshold,self.high_threshold)
    
    def connectedCameraCount(self,max_camera_count = 5):
        self.CAMERA_INDICES = []
        _NUMBER_OF_CONNECTED_CAMERAS = 0
        print('')
        print(f'{Fore.BLUE}#####################################################{Style.RESET_ALL}')
        print(f'{Fore.RED}connectedCameraCount(){Style.RESET_ALL} -> Function is being {Fore.RED}executed{Style.RESET_ALL}!')
        print('Start time of module execution: '+ str(datetime.datetime.now()))
        print('')
        print('Checking Number Of Connected Cameras')
        print(str(max_camera_count)+' ports will be tested.')
        print('')
        pbar = ProgressBar(widgets = ['Testing Camera Ports: ',SimpleProgress(),
                                      ' ', Bar(),
                                      ' ', Percentage(),
                                      ' ', Timer(),
                                      ' ', AdaptiveETA(),
                                      ], maxval = max_camera_count)
                
        for i in range(max_camera_count):
            cap = cv2.VideoCapture(i)
            control,_ = cap.read()
            cap.release()
            if control is True:
                self.CAMERA_INDICES.append(i)
            if control is False:
                num_of_connected_cameras = i
                break
            pbar.update(i+1)
            
        pbar.finish()
        print('')
        print(str(max_camera_count)+' camera ports were tested.')
        print(str(num_of_connected_cameras)+' cameras are present.')
        print('')
        print(f'{Fore.GREEN}#####################################################{Style.RESET_ALL}')
        del i,control
        self._NUMBER_OF_CONNECTED_CAMERAS = num_of_connected_cameras
        if _NUMBER_OF_CONNECTED_CAMERAS is not 0:
            self.isCameraDetected = True
        return num_of_connected_cameras
    
    def detectExternalCamera(self):
        self._EXTERNAL_CAMERA_INDEX_LIST_ = []
        if self._NUMBER_OF_CONNECTED_CAMERAS is None:
            print('\nError detectExternalCamera(). Either first run connectedCameraCount(), ' +
                  'or import calibration parameters directly.\n')
        else:
            print('')
            print(f'{Fore.BLUE}#####################################################{Style.RESET_ALL}')
            print(f'{Fore.RED}detectExternalCamera(){Style.RESET_ALL} -> Function is being {Fore.RED}executed{Style.RESET_ALL}!')
            print('Start time of module execution: '+ str(datetime.datetime.now()))
            print('')
            print('Checking External Cameras. Make sure built-in camera is covered or disabled.')
            print('')
            pbar = ProgressBar(widgets = ['Camera: ',SimpleProgress(),
                                          ' ', Timer()
                                          ],minval=1, maxval = self._NUMBER_OF_CONNECTED_CAMERAS)
            pbar_2 = ProgressBar(widgets = ['Testing Camera Ports: ',SimpleProgress(),
                                          ' ', Bar(),
                                          ' ', Percentage(),
                                          ' ', Timer(),
                                          ' ', AdaptiveETA(),
                                          ], maxval = self._NUMBER_OF_FRAMES_FOR_EXTERNAL_CAMERA_DETECTION)
                
        for i,camera_id in enumerate(self.CAMERA_INDICES):
            append_array = np.array([], dtype=np.int32).reshape(0,self.CameraDIM[0],self.CameraDIM[1])
            cap = cv2.VideoCapture(camera_id)
            
            for j in range(self._NUMBER_OF_FRAMES_FOR_EXTERNAL_CAMERA_DETECTION):
                status = False
                while status == False:
                    status,frame = cap.read()
                frame = self._bgr2gray(frame)
                frame = frame.reshape(1,self.CameraDIM[0],self.CameraDIM[1]).astype(np.uint8)
                append_array = np.vstack([append_array, frame])
                pbar_2.update(j+1)
                
            cap.release()
            mean_append_array = np.mean(append_array, axis=0)
            mean_append_array_2 = np.mean(mean_append_array).astype(np.uint8)
            print('')
            print('Average intensity is:')
            print(mean_append_array_2)
            if mean_append_array_2 > self._ELEMINATION_THRESHOLD_:
                self._EXTERNAL_CAMERA_INDEX_LIST_.append(i)
                
            pbar.update(i+1)
            
        pbar.finish()
        pbar_2.finish()
        print('')
        print('Following external camera ports were found.')
        print (', '.join(str(index) for index in self._EXTERNAL_CAMERA_INDEX_LIST_))
        print('If you believe error occured, change _ELEMINATION_THRESHOLD_')
        print('')
        print(f'{Fore.GREEN}#####################################################{Style.RESET_ALL}')
        if len(self._EXTERNAL_CAMERA_INDEX_LIST_) is not 0:
            self.isExternalCameraDetecred = True
        #del frame, camera_id, i, j, append_array, mean_append_array, mean_append_array_2
        return self._EXTERNAL_CAMERA_INDEX_LIST_
            
        
    def initializeExternalCamera(self):
        isCameraOpen_list = []
        isImageCollected_list = []
        print('')
        print(f'{Fore.BLUE}#####################################################{Style.RESET_ALL}')
        print(f'{Fore.RED}initializeExternalCamera(){Style.RESET_ALL} -> Function is being {Fore.RED}executed{Style.RESET_ALL}!')
        print('Start time of module execution: '+ str(datetime.datetime.now()))
        print('')
        print('Checking Number Of Connected Cameras')
        if self._EXTERNAL_CAMERA_INDEX_LIST_ is not None and len(self._EXTERNAL_CAMERA_INDEX_LIST_) is not 0:
            if len(self.CAMERA_LIST) is not 0:
                print('')
                print('Re-initialization detected. First releasing External Cameras.')
                release_bool = self.releaseExternalCamera()
                if release_bool is True:
                    print('Release Completed.')
                else:
                    print('ERROR OCCURED')
                    raise ValueError('Could not released Cameras in initializeExternalCamera module')
                self.CAMERA_LIST = []
                print('Camera Object list emptied.')
                print('Reinitializing...')
            else:
                print('')
                print('Initializing...')
                print('')
                
            for index,value in enumerate(self._EXTERNAL_CAMERA_INDEX_LIST_):
                self.CAMERA_LIST.append(cv2.VideoCapture(value))
                print('Camera '+ str(index+1)+' with port ID '+ str(value)+ ' is initialized.')
                print('Waiting for camera stabilization.')
                start_time = datetime.datetime.now().timestamp()
                current_time = start_time
                while(current_time-start_time < 1.0):
                    current_time = datetime.datetime.now().timestamp()
                    
                camera_open_bool = self.CAMERA_LIST[index].isOpened()
                isCameraOpen_list.append(camera_open_bool)
                print('Is camera '+str(value+1)+' open: '+str(camera_open_bool))
                print('Checking if image can be captured with initialized camera...')
                condition,_ = self.CAMERA_LIST[index].read()
                isImageCollected_list.append(condition)
                print('Image Captured: '+str(condition))
                print('')
            isCameraOpen_bool = True
            for index,value in enumerate(isCameraOpen_list):
                if value is False:
                    isCameraOpen_bool = False
            isImageCollected_bool = True
            for index,value in enumerate(isImageCollected_list):
                if value is False:
                    isImageCollected_bool = False
         
            if (isImageCollected_bool is True) and (isCameraOpen_bool is True):
                print('Everything is OKAY if you did not see any FALSE on terminal.')
                print('Cameras can be accesed with image.CAMERA_LIST[i]')
            else:
                print('Something is wrong with cameras...')
                print('isCameraOpen_bool: '+ str(isCameraOpen_bool))
                print('isImageCollected_bool: '+ str(isImageCollected_bool))
                raise ValueError('Either camera did not open, or image could not collected properly.')
                
            print('')
            print(f'{Fore.GREEN}#####################################################{Style.RESET_ALL}')
            self.isCameraInitialized = True
            self.isCameraOpen = True
            return True
        else:
            print('Use detectExternalCamera() first. Because self._EXTERNAL_CAMERA_INDEX_LIST_ = None')
            print('')
            print(f'{Fore.GREEN}#####################################################{Style.RESET_ALL}')
            raise ValueError('self.CAMERA_LIST = None')
            return False
        
        
        
    def releaseExternalCamera(self):
        print('')
        print(f'{Fore.BLUE}#####################################################{Style.RESET_ALL}')
        print(f'{Fore.RED}releaseExternalCamera(){Style.RESET_ALL} -> Function is being {Fore.RED}executed{Style.RESET_ALL}!')
        print('Start time of module execution: '+ str(datetime.datetime.now()))
        print('')
        if self.CAMERA_LIST is None:
            raise ValueError('As you might know, in order to release camera, you have to first initialize it...')
        isCamerasClosed_list = []
        isCamerasClosed_bool = True
        for index,value in enumerate(self.CAMERA_LIST):
            print('Camera '+ str(index+1)+' with object ID '+ str(value)+ ' will be closed.')
            camera_open_bool = self.CAMERA_LIST[index].isOpened()
            if camera_open_bool is True:
                self.CAMERA_LIST[index].release()
            camera_open_bool = self.CAMERA_LIST[index].isOpened() 
            isCamerasClosed_list.append(camera_open_bool)
            print('Is Camera '+str(index+1)+' open: '+str(camera_open_bool))
            print('')
        for index,value in enumerate(isCamerasClosed_list):
            if value is True:
                isCamerasClosed_bool = False
        if isCamerasClosed_bool is False:
            raise ValueError('Error occured during camera closing.')
            return False
        else:
            print('YES! All cameras are closed properly.')
            self.isCameraOpen = False
        print('')
        print(f'{Fore.GREEN}#####################################################{Style.RESET_ALL}')    
        return True
            
    def readExternalCamera(self,):
        self.IMAGE_LIST = []
        for index,value in enumerate(self.CAMERA_LIST):
            self.IMAGE_LIST.append(value.read())
        #read as LIST[0][0] for read status
        #read asvLIST[0][1] for image array
        return self.IMAGE_LIST

        #This function will not print any value since it will slow system dramatically.
        
    def setCameraMatrix(self, CameraMatrix = None):
        if CameraMatrix is None:
            print('Error setting CameraMatrix. No Matrix supplied.')
            return False
        else:
            self.CameraMatrix = CameraMatrix
            return True
        
    def getCameraMatrix(self):
        return self.CameraMatrix
    
    def setDistorsionCoefficients(self, DistorsionCoefficients = None):
        if DistorsionCoefficients is None:
            print('Error setting DistorsionCoefficients. No Matrix supplied.')
            return False
        else:
            self.DistorsionCoefficients = DistorsionCoefficients
            return True
        
    def getDistorsionCoefficients(self):
        return self.DistorsionCoefficients

    def setCameraDIM(self, DIM = None):
        if DIM is None:
            print('Error setting CameraDIM. No Tuple supplied.')
            return False
        else:
            self.CameraDIM = DIM
            return True
        
    def getCameraDIM(self):
        return self.CameraDIM
    
    def setCameraFPS(self, FPS = None):
        if FPS is None:
            print('Error setting CameraFPS. No int supplied.')
            return False
        else:
            self.CameraFPS = FPS
            return True
        
    def getCameraFPS(self):
        return self.CameraFPS
    
    def calibrateParameters(self):
        pass
    
    def readParameters(self):
        pass
    
    def setParameters(self):
        pass

    #Private functions to convert color from BGR colorspace
    def _bgr2gray(self,img_arr):
        """
        BGR to GRAYSCALE
        :param img_arr: Input as 3 DIM NUMPY array
        :return: Output as 2 DIM NUMPY array
        """
        return cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)

    def _bgr2rgb(self,img_arr):
        """
        BGR to RGB
        :param img_arr: Input as 3 DIM NUMPY array
        :return: Output as 3 DIM NUMPY array
        """
        return cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)

    def _bgr2hsv(self,img_arr):
        """
        BGR to HSV
        :param img_arr: Input as 3 DIM NUMPY array
        :return: Output as 3 DIM NUMPY array
        """
        return cv2.cvtColor(img_arr, cv2.COLOR_BGR2HSV)

    def _bgr2lab(self,img_arr):
        """
        BGR to Lab
        :param img_arr: Input as 3 DIM NUMPY array
        :return: Output as 3 DIM NUMPY array
        """
        return cv2.cvtColor(img_arr, cv2.COLOR_BGR2LAB)

    def _bgr2YCrCb(self,img_arr):
        """
        BGR to YCrCb
        :param img_arr: Input as 3 DIM NUMPY array
        :return: Output as 3 DIM NUMPY array
        """
        return cv2.cvtColor(img_arr, cv2.COLOR_BGR2YCrCb)


    #Private functions to convert color from RGB colorspace
    def _rgb2gray(self,img_arr):
        """
        RGB to GRAYSCALE
        :param img_arr: Input as 3 DIM NUMPY array
        :return: Output as 2 DIM NUMPY array
        """
        return cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)

    def _rgb2bgr(self,img_arr):
        """
        RGB to BGR
        :param img_arr: Input as 3 DIM NUMPY array
        :return: Output as 3 DIM NUMPY array
        """
        return cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)

    def _rgb2hsv(self,img_arr):
        """
        RGB to HSV
        :param img_arr: Input as 3 DIM NUMPY array
        :return: Output as 3 DIM NUMPY array
        """
        return cv2.cvtColor(img_arr, cv2.COLOR_RGB2HSV)

    def _rgb2lab(self,img_arr):
        """
        RGB to Lab
        :param img_arr: Input as 3 DIM NUMPY array
        :return: Output as 3 DIM NUMPY array
        """
        return cv2.cvtColor(img_arr, cv2.COLOR_RGB2LAB)

    def _rgb2YCrCb(self,img_arr):
        """
        RGB to YCrCb
        :param img_arr: Input as 3 DIM NUMPY array
        :return: Output as 3 DIM NUMPY array
        """
        return cv2.cvtColor(img_arr, cv2.COLOR_RGB2YCrCb)

    #Private functions to convert color from HSV colorspace
    def _hsv2gray(self,img_arr):
        """
        HSV to GRAYSCALE
        :param img_arr: Input as 3 DIM NUMPY array
        :return: Output as 2 DIM NUMPY array
        """
        return cv2.cvtColor(img_arr, cv2.COLOR_HSV2GRAY)

    def _hsv2bgr(self,img_arr):
        """
        HSV to BGR
        :param img_arr: Input as 3 DIM NUMPY array
        :return: Output as 3 DIM NUMPY array
        """
        return cv2.cvtColor(img_arr, cv2.COLOR_HSV2BGR)

    def _hsv2rgb(self,img_arr):
        """
        HSV to RGB
        :param img_arr: Input as 3 DIM NUMPY array
        :return: Output as 3 DIM NUMPY array
        """
        return cv2.cvtColor(img_arr, cv2.COLOR_HSV2RGB)

    def _hsv2lab(self,img_arr):
        """
        HSV to Lab
        :param img_arr: Input as 3 DIM NUMPY array
        :return: Output as 3 DIM NUMPY array
        """
        return cv2.cvtColor(img_arr, cv2.COLOR_HSV2LAB)

    def _hsv2YCrCb(self,img_arr):
        """
        HSV to YCrCb
        :param img_arr: Input as 3 DIM NUMPY array
        :return: Output as 3 DIM NUMPY array
        """
        return cv2.cvtColor(img_arr, cv2.COLOR_HSV2YCrCb)

    #Private functions to convert color from LAB colorspace
    def _lab2gray(self,img_arr):
        """
        Lab to GRAYSCALE
        :param img_arr: Input as 3 DIM NUMPY array
        :return: Output as 2 DIM NUMPY array
        """
        return cv2.cvtColor(img_arr, cv2.COLOR_LAB2GRAY)

    def _lab2bgr(self,img_arr):
        """
        Lab to BGR
        :param img_arr: Input as 3 DIM NUMPY array
        :return: Output as 3 DIM NUMPY array
        """
        return cv2.cvtColor(img_arr, cv2.COLOR_LAB2BGR)

    def _lab2rgb(self,img_arr):
        """
        Lab to RGB
        :param img_arr: Input as 3 DIM NUMPY array
        :return: Output as 3 DIM NUMPY array
        """
        return cv2.cvtColor(img_arr, cv2.COLOR_LAB2RGB)

    def _lab2hsv(self,img_arr):
        """
        Lab to HSV
        :param img_arr: Input as 3 DIM NUMPY array
        :return: Output as 3 DIM NUMPY array
        """
        return cv2.cvtColor(img_arr, cv2.COLOR_LAB2HSV)

    def _lab2YCrCb(self,img_arr):
        """
        Lab to YCrCb
        :param img_arr: Input as 3 DIM NUMPY array
        :return: Output as 3 DIM NUMPY array
        """
        return cv2.cvtColor(img_arr, cv2.COLOR_LAB2YCrCb)


    #Private functions to convert color from YCrCb colorspace
    def _YCrCb2gray(self,img_arr):
        """
        YCrCb to GRAYSCALE
        :param img_arr: Input as 3 DIM NUMPY array
        :return: Output as 2 DIM NUMPY array
        """
        return cv2.cvtColor(img_arr, cv2.COLOR_YCrCb2GRAY)
    def _YCrCb2bgr(self,img_arr):
        """
        YCrCb to BGR
        :param img_arr: Input as 3 DIM NUMPY array
        :return: Output as 3 DIM NUMPY array
        """
        return cv2.cvtColor(img_arr, cv2.COLOR_YCrCb2BGR)
    def _YCrCb2rgb(self,img_arr):
        """
        YCrCb to RGB
        :param img_arr: Input as 3 DIM NUMPY array
        :return: Output as 3 DIM NUMPY array
        """
        return cv2.cvtColor(img_arr, cv2.COLOR_YCrCb2RGB)
    def _YCrCb2hsv(self,img_arr):
        """
        YCrCb to HSV
        :param img_arr: Input as 3 DIM NUMPY array
        :return: Output as 3 DIM NUMPY array
        """
        return cv2.cvtColor(img_arr, cv2.COLOR_YCrCb2HSV)
    def _YCrCb2lab(self,img_arr):
        """
        YCrCb to Lab
        :param img_arr: Input as 3 DIM NUMPY array
        :return: Output as 3 DIM NUMPY array
        """
        return cv2.cvtColor(img_arr, cv2.COLOR_YCrCb2LAB)

    def _histogramNormalize(self,img_arr, input_type = 'hsv'):
        """
        Private function to convert normalize value channel of HSV. After histogram normalization, image in no longer
        sensitive to lighting conditions.
        :param img_arr: Input NDIM NUMPY array.
        :param input_type: Input colorspace as STRING.
        :return: returns NUMPY array with input's NDIM.
        """
        self.normalizeInputType = input_type
        ndim = img_arr.ndim

        #if bgr, convert to hsv, equalize 3rd channel (value), return to original input format
        if (self.normalizeInputType == 'bgr' and ndim == 3):
            img_arr = self._bgr2hsv(img_arr)
            img_arr[:,:,2] = exposure.equalize_hist(img_arr[:,:,2])
            self.normalizeInputType = 'hsv'
            return self._hsv2bgr(img_arr)

        #if hsv, equalize 3rd channel (value), return to original input format
        elif (self.normalizeInputType == 'hsv' and ndim == 3):
            img_arr[:,:,2] = exposure.equalize_hist(img_arr[:,:,2])
            self.normalizeInputType = 'hsv'
            return img_arr

        #if rgb, convert to hsv, equalize 3rd channel (value), return to original input format
        elif (self.normalizeInputType == 'rgb' and ndim == 3):
            img_arr = self._rgb2hsv(img_arr)
            img_arr[:,:,2] = exposure.equalize_hist(img_arr[:,:,2])
            self.normalizeInputType = 'hsv'
            return self._hsv2rgb(img_arr)

        #if YCrCb, convert to hsv, equalize 3rd channel (value), return to original input format
        elif (self.normalizeInputType == 'ycrbb' and ndim == 3):
            img_arr = self._YCrCb2hsv(img_arr)
            img_arr[:,:,2] = exposure.equalize_hist(img_arr[:,:,2])
            self.normalizeInputType = 'hsv'
            return self._hsv2YCrCb(img_arr)

        #if lab, convert to hsv, equalize 3rd channel (value), return to original input format
        elif (self.normalizeInputType == 'lab' and ndim == 3):
            img_arr = self._lab2hsv(img_arr)
            img_arr[:,:,2] = exposure.equalize_hist(img_arr[:,:,2])
            self.normalizeInputType = 'hsv'
            return self._hsv2lab(img_arr)

        #if gray, equalize channel (value), return to original input format
        elif (self.normalizeInputType == 'gray' and ndim == 2):
            self.normalizeInputType = 'hsv'
            img_arr = exposure.equalize_hist(img_arr)
            return img_arr

        #if bgr, convert to hsv, equalive 3rd channel (value), return to original input format
        else:
            self.normalizeInputType = 'hsv'
            raise ValueError('Given ARRAY\'s NDIM does not match INPUT_TYPE\'s NDIM. Check uses of this module.')

    @property
    def __version__(self):
        print(self._version)

    @property
    def __author__(self):
        print('Author: '+self._author)

    @property
    def __cv2version__(self):
        print(self._cv2version)

