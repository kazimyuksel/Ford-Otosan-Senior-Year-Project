# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 14:15:31 2018

@author: Kazım
"""
import cv2
import time
import numpy as np

DIM = (640,480)
K=np.array([[434.69582589661127, 0.0, 329.88593484416015], [0.0, 432.9282346935097, 218.16156689618913], [0.0, 0.0, 1.0]])
D=np.array([[-0.08421582695848977], [0.1072058128379107], [-0.5622664731446513], [0.42955908638253876]])
K=np.array([[433.1789201825628, 0.0, 336.0326551869997], [0.0, 430.52735997037956, 216.5706380556904], [0.0, 0.0, 1.0]])
D=np.array([[-0.05280848062554133], [-0.13174782579340452], [-0.028567480262252885], [0.060944836127339065]])

dim1 = DIM
dim2 = (640, 480)
dim3 = (640, 480)
balance=1.0

scaled_K = K * dim1[0] / DIM[0]  # The values of K is to scale with image dimension.
scaled_K[2][2] = 1.0  # Except that K[2][2] is always 1.0
# This is how scaled_K, dim2 and balance are used to determine the final K used to un-distort image. OpenCV document failed to make this clear!
new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, D, dim2, np.eye(3), balance=balance)
map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, D, np.eye(3), new_K, dim3, cv2.CV_16SC2)

##################
DELAY = 0.02
USE_CAM = 1
IS_FOUND = 0

MORPH = 7
CANNY = 250
##################
# 420x600 oranı 105mmx150mm gerçek boyuttaki kağıt için
_width  = 30.0
_height = 30.0
_margin = 470.0

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

lower_green = np.array([45,40,40])
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
		rgb = cv2.bilateralFilter( rgb, 9, 50, 50 )
		gaussian_3 = cv2.GaussianBlur(rgb, (3,3), 10.0)
		rgb = cv2.addWeighted(rgb, 1.2, gaussian_3, -0.2, 0, rgb)
		rgb = cv2.remap(rgb, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
		gaussian_3 = cv2.GaussianBlur(rgb, (3,3), 10.0)
		rgb = cv2.addWeighted(rgb, 1.5, gaussian_3, -0.5, 0, rgb)
		rgb = cv2.bilateralFilter( rgb, 9, 20, 20 )
		gaussian_3 = cv2.GaussianBlur(rgb, (3,3), 10.0)
		rgb = cv2.addWeighted(rgb, 1.2, gaussian_3, -0.2, 0, rgb)
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
		closed = cv2.bilateralFilter( closed, 9, 75, 75 ) + mask
		closed = cv2.bilateralFilter( closed, 1, 10, 120 )
		cv2.imshow( 'closed', closed )

		_,contours, h = cv2.findContours( closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE )

		for cont in contours:

			# Küçük alanları pass geç
			if cv2.contourArea( cont ) > 100 :

				arc_len = cv2.arcLength( cont, True )

				approx = cv2.approxPolyDP( cont, 0.01 * arc_len, True )

				if ( len( approx ) == 4 ):
					IS_FOUND = 1
					#M = cv2.moments( cont )
					#cX = int(M["m10"] / M["m00"])
					#cY = int(M["m01"] / M["m00"])
					#cv2.putText(rgb, "Center", (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

					pts_src = np.array( approx, np.float32 )

					h, status = cv2.findHomography( pts_src, pts_dst )
					out = cv2.warpPerspective( temp, h, ( int( _width + _margin * 2 ), int( _height + _margin * 2 ) ) )


					cv2.drawContours( temp, [approx], -1, ( 255, 0, 0 ), 2 )

				else : pass

		cv2.imshow( 'closed', closed )
		cv2.imshow( 'gray', gray )

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