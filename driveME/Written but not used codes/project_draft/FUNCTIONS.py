# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 15:26:12 2019

@author: Kazım
"""


import cv2
from colorama import Fore, Style
import numpy as np
import progressbar
from progressbar import ProgressBar, Bar, AdaptiveETA, Percentage, SimpleProgress, Timer
import datetime

def connectedCameraCount(max_camera_count = 5):
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
    for i in range(5):
        cap = cv2.VideoCapture(i)
        control,_ = cap.read()
        cap.release()
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
    return num_of_connected_cameras

_NUMBER_OF_CONNECTED_CAMERAS_ = connectedCameraCount()


def startTime():
    print('')
    print(f'{Fore.BLUE}#####################################################{Style.RESET_ALL}')
    print(f'{Fore.RED}startTime(){Style.RESET_ALL} -> Function is being {Fore.RED}executed{Style.RESET_ALL}!')
    print('Start time of module execution: '+ str(datetime.datetime.now()))
    print('')
    print('Start time of MAIN execution: '+ str(datetime.datetime.now()))
    print('')
    print(f'{Fore.GREEN}#####################################################{Style.RESET_ALL}')
    return (np.uint32)(datetime.datetime.now().replace(microsecond = 0).timestamp())
    
_START_TIMESTAMP_ = startTime()

def testRunningModuleVersion():
    _NP_VERSION_REQUIRED_ = '1.15.4'
    _PROGRESSBAR_VERSION_REQUIRED_ = '3.39.2'
    _OPENCV_VERSION_REQUIRED_ = '3.4.4'
    print('')
    print(f'{Fore.BLUE}#####################################################{Style.RESET_ALL}')
    print(f'{Fore.RED}testRunningModuleVersion(){Style.RESET_ALL} -> Function is being {Fore.RED}executed{Style.RESET_ALL}!')
    print('Start time of module execution: '+ str(datetime.datetime.now()))
    print('')
    print('NumPy version :\t\t' + np.__version__ +'\t|   '+_NP_VERSION_REQUIRED_+' or up required.\n'
          'progressbar version :\t'+ progressbar.__version__ + '\t|   '+_PROGRESSBAR_VERSION_REQUIRED_+' or up required. \n'
          'OpenCV version :\t'+ cv2.__version__ + '\t|   '+_OPENCV_VERSION_REQUIRED_+' or up required.')
    print('')
    print(f'{Fore.GREEN}#####################################################{Style.RESET_ALL}')
    
testRunningModuleVersion()


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    