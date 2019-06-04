# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 16:58:16 2018

@author: Kazım
"""

import cv2
import numpy as np
import math
from imutils.video import WebcamVideoStream
from imutils.video import FPS
import argparse
import imutils



def localizeSign(frame):

   hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

   # define range of blue color in HSV
   lower_red = np.array([160,140,50])
   upper_red = np.array([180,255,255])

   imgThreshHigh = cv2.inRange(hsv, lower_red, upper_red)
   thresh = imgThreshHigh.copy()

   _, countours,_ = cv2.findContours(thresh, cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
  
   max_area=0
   best_cnt=0
  
   for cnt in countours:
      
       try:
           area = cv2.contourArea(cnt)
          
       except:
          
           area=math.pi
          
       if area > max_area:
          
           max_area = area
           best_cnt = cnt

   M = cv2.moments(best_cnt)
  
   if(M['m00']>10e-14):
       cx,cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
   else:
       #cx,cy=(None,None)
       cx,cy=(50,50)
   #coord = cx, cy #This are your coordinates for the circle
   try:
       area = cv2.contourArea(best_cnt) #save the object area
   except:
       area=math.pi
   #perimeter = cv2.arcLength(best_cnt,True) is the object perimeter

   #Save the coords every frame on a list
   #Here you can make more conditions if you don't want repeated coordinate
   return {'Tresh': thresh, 'C_x': cx, 'C_y': cy, 'Area': area}

stream = cv2.VideoCapture(1)
stream.set(3,640)
stream.set(4,480)

while True:
	# grab the frame from the stream and resize it to have a maximum
	# width of 400 pixels
	(grabbed, frame) = stream.read()
    
	output = localizeSign(frame)
	radius = np.sqrt(output['Area']).astype(np.uint8)
	cx = output['C_x']
	cy = output['C_y']
	cv2.circle(frame,(cx,cy), radius, (0,0,255), -1)
	# check to see if the frame should be displayed to our screen
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	if (key == ord("q")):
		print("Exiting")
		stream.release()
		cv2.destroyAllWindows()
		break


