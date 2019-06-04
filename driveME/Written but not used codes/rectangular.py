# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 11:31:36 2018

@author: Kazım
"""

#-*- coding: utf-8 -*-
# based on https://mehmethanoglu.com.tr/blog/6-opencv-ile-dikdortgen-algilama-python.html
import cv2
import time
import numpy as np
from skimage import exposure, color
from skimage import data, img_as_float

"""
sudo apt-get install python-opencv
sudo apt-get install python-matplotlib
"""

##################
DELAY = 0.02
USE_CAM = 1
IS_FOUND = 0

MORPH = 7
CANNY = 250
##################
# 420x600 oranı 105mmx150mm gerçek boyuttaki kağıt için
_width  = 500.0
_height = 500.0
_margin = 100.0

empty = 'top_left'
##################

if USE_CAM:
	video_capture = cv2.VideoCapture(0)

corners = np.array(
	[
		[[  		_margin, _margin 			]],
		[[ 			_margin, _height + _margin  ]],
		[[ _width + _margin, _height + _margin  ]],
		[[ _width + _margin, _margin 			]],
	]
)

pts_dst = np.array( corners, np.float32 )

lower_green = np.array([45,50,50])
upper_green = np.array([75,255,255])

lower_red_1 = np.array([0,50,50])
upper_red_1 = np.array([25,255,255])
lower_red_2 = np.array([230,50,50])
upper_red_2 = np.array([255,255,255])

lower_blue = np.array([100,50,50])
upper_blue = np.array([140,255,255])

lower_magenta = np.array([125,50,50])
upper_magenta = np.array([175,255,255])

while True :

	if USE_CAM :
		ret, rgb = video_capture.read()
	else :
		ret = 1
		rgb = cv2.imread( "D:\\yamuk2.jpg", 1 )

	if ( ret ):
		temp = rgb
		hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
		hsv = hsv.astype(np.uint8)
        
		mask = cv2.inRange(hsv, lower_green, upper_green)	
		rgb = cv2.bitwise_and(rgb,rgb, mask= mask) 
		rgb = rgb.astype(np.uint8)
		mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
		mask_red_1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
		mask_red_2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
		mask_magenta = cv2.inRange(hsv, lower_magenta, upper_magenta)
		
		blue = cv2.bitwise_and(temp,temp, mask= mask_blue)
		red_1 = cv2.bitwise_and(temp,temp, mask= mask_red_1)
		red_2 = cv2.bitwise_and(temp,temp, mask= mask_red_2)
		magenta = cv2.bitwise_and(temp,temp, mask= mask_magenta)
		
		total_patern_mask = blue + red_1 + red_2 + magenta + rgb

		gray = cv2.cvtColor( rgb, cv2.COLOR_BGR2GRAY )

		gray = cv2.bilateralFilter( gray, 9, 75, 75 )
		otsu, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY or cv2.THRESH_OTSU)
		low_threshold = otsu * 0.5
		high_threshold = otsu
		edges = cv2.Canny(gray,low_threshold,high_threshold)

		#edges  = cv2.Canny( gray, 10, CANNY )

		kernel = cv2.getStructuringElement( cv2.MORPH_RECT, ( MORPH, MORPH ) )
		closed = cv2.morphologyEx( edges, cv2.MORPH_CLOSE, kernel )
		closed = cv2.bilateralFilter( closed, 9, 75, 75 )
		cv2.imshow( 'closed', closed )

		_,contours, h = cv2.findContours( closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE )

		for cont in contours:

			# Küçük alanları pass geç
			if cv2.contourArea( cont ) > 1000 :

				arc_len = cv2.arcLength( cont, True )

				approx = cv2.approxPolyDP( cont, 0.1 * arc_len, True )

				if ( len( approx ) == 4 ):
					IS_FOUND = 1
					#M = cv2.moments( cont )
					#cX = int(M["m10"] / M["m00"])
					#cY = int(M["m01"] / M["m00"])
					#cv2.putText(rgb, "Center", (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

					pts_src = np.array( approx, np.float32 )

					h, status = cv2.findHomography( pts_src, pts_dst )
					out = cv2.warpPerspective( total_patern_mask, h, ( int( _width + _margin * 2 ), int( _height + _margin * 2 ) ) )


					cv2.drawContours( total_patern_mask, [approx], -1, ( 255, 0, 0 ), 2 )

				else : pass

		#cv2.imshow( 'closed', closed )
		#cv2.imshow( 'gray', gray )

		cv2.imshow( 'edges', edges )


		cv2.imshow( 'original', temp )
		cv2.imshow( 'pattern', total_patern_mask )

		if IS_FOUND :

			cv2.imshow( 'out', out )

		if cv2.waitKey(27) & 0xFF == ord('q') :
			break

		if cv2.waitKey(99) & 0xFF == ord('c') :
			current = str( time.time() )
			cv2.imwrite( 'ocvi_' + current + '_edges.jpg', edges )
			cv2.imwrite( 'ocvi_' + current + '_gray.jpg', gray )
			cv2.imwrite( 'ocvi_' + current + '_org.jpg', rgb )
			print ("Pictures saved")

		time.sleep( DELAY )

	else :
		print ("Stopped")
		break

if USE_CAM : video_capture.release()
cv2.destroyAllWindows()

# end

#cv2.imshow( 'edges', edges )
#cv2.imshow( 'dst', dst )
#cv2.imshow( 'mask', mask )
#cv2.imshow( 'gray', gray )
#cv2.imshow( 'closed', closed )
#cv2.waitKey(0)
#cv2.destroyAllWindows()