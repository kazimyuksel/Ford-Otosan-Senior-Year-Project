import os
import cv2
from colorama import Fore, Style
import numpy as np
import progressbar
from progressbar import ProgressBar, Bar, AdaptiveETA, Percentage, SimpleProgress, Timer
import datetime
from image.ImageProcess import ImageProcess

#First time run or not?
IS_CALIBRATION_REQUIRED = True
_RUNNING_DIRECTORY = 'D:\FordOtosan'
if not os.path.exists(_RUNNING_DIRECTORY):
    print('')
    print(f'{Fore.BLUE}#####################################################{Style.RESET_ALL}')
    print('Start time of module execution: '+ str(datetime.datetime.now()))
    print(f'{Fore.RED}REQUIRED FOLDER(){Style.RESET_ALL} '+_RUNNING_DIRECTORY+'{Fore.RED}DOES NOT EXIST!{Style.RESET_ALL}!')
    print('Please create folder and try again.')
    print('')

if IS_CALIBRATION_REQUIRED == True:
    calibration()
else:
    _imported_parameters = load_json_parameters()
    ASSIGN PARAMETERS
    del _imported_parameters
    
    
#Get Starting Time
_START_TIMESTAMP = startTime()

#Test Versions of Imported Modules
testRunningModuleVersion()

#Initialize ImageProcess() class
image = ImageProcess()

#Test Ports for Camera Connection
_NUMBER_OF_CONNECTED_CAMERAS = image.connectedCameraCount()

#Test Cameras to Identify Built-in Camera
#This method requires user action.
#This method may require covering of built-in camera to test exposure
#Keep this in mind!
_EXTERNAL_CAMERA_INDEX = image.detectExternalCamera()

#Initialize detected cameras.
#This class method does not take camera index as input, it already stores it
#in its self.method.
image.initializeExternalCamera(height=480, width=640, fps=20)



