# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 01:30:39 2019

@author: Kazım
"""

occp_map = cv2.imread("occp.png",0)
print(np.sum(cv2.bitwise_not(occp_map))/255)
occp_map=cv2.adaptiveThreshold(occp_map,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,3,1)
print(np.sum(cv2.bitwise_not(occp_map))/255)
imgShow(occp_map)
cv2.imwrite("occp_outer.png",occp_map)