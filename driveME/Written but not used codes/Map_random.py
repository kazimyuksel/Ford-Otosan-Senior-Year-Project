# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 15:14:17 2019

@author: Kazım
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 20:44:30 2019

@author: Kazım
"""
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
kernel = np.ones((3,3), np.uint8) 
image = cv2.imread("D:\\FordOtosan\\occupancy.PNG",0)
#image = cv2.imread("D:\\_ravel.PNG",0)
#threshold image to get binary image
_,source_image = cv2.threshold(image,120,255,cv2.THRESH_BINARY)
img_erosion = cv2.erode(source_image, kernel, iterations=1) 
dist = cv2.distanceTransform(img_erosion, cv2.DIST_L2, 5)
pdf = dist / np.sum(dist);
#dist = np.power(dist,0.9)
dist_1d = pdf.ravel()
# Normalize the distance image for range = {0.0, 1.0}
# so we can visualize and threshold it
cv2.normalize(dist, dist, 0, 255.0, cv2.NORM_MINMAX)

cv2.imshow('Distance Transform Image', dist.astype(np.uint8))
#dist = cv2.GaussianBlur(dist,(7,7),5)
#cv2.imshow('Distance Transform Image BLUR', dist)
cv2.imshow('eroded', img_erosion)
a = np.cumsum(dist_1d)
u, indices = np.unique(a, return_index=True)
u = u[1:]
indices = indices[1:]
#elems = np.arange(0, a.shape[0], 1)
#plt.scatter(elems,a)
#plt.show()
#


#rand_point=np.random.random_sample()
#def find_nearest(array_index,array_cdf, value):
#    array_cdf = np.asarray(array_cdf)
#    idx = (np.abs(array_cdf - value)).argmin()
#    return array_index[idx]
#
#
#initial_array = np.zeros(dist.shape)
#while True:
#    rand_point=np.random.random_sample()
#    #print(find_nearest(indices,u, rand_point))
#    nearest_index = find_nearest(indices,u, rand_point)
#    x = np.mod(nearest_index,dist.shape[1])
#    y = int(nearest_index/dist.shape[1])
#    initial_array[y,x] = initial_array[y,x] + 100
#    if initial_array[y,x] > 255:
#        initial_array[y,x] = 255
#    cv2.imshow('eroded', initial_array.astype(np.uint8))
#    if cv2.waitKey(27) & 0xFF == ord('q') :
#        break
#cv2.waitKey(0)
#cv2.destroyAllWindows()

def initGaussianKernel(shape,sigma1,sigma2):
    kernel_edge = 2*max(shape)
    gaussianKernal_1D_row = cv2.getGaussianKernel(kernel_edge,sigma1)
    gaussianKernal_1D_col = cv2.getGaussianKernel(kernel_edge,sigma1)

    gaussianKernal_2D = gaussianKernal_1D_row*gaussianKernal_1D_col.T
    gaussianKernal_2D = (gaussianKernal_2D-gaussianKernal_2D.min())/(gaussianKernal_2D.max()-gaussianKernal_2D.min())
    
    gaussianKernal_1D_row = cv2.getGaussianKernel(kernel_edge,sigma2)
    gaussianKernal_1D_col = cv2.getGaussianKernel(kernel_edge,sigma2)
    
    gaussianKernal_2D_1 = gaussianKernal_1D_row*gaussianKernal_1D_col.T
    gaussianKernal_2D_1 = (gaussianKernal_2D_1-gaussianKernal_2D_1.min())/(gaussianKernal_2D_1.max()-gaussianKernal_2D_1.min())

    bandpass = gaussianKernal_2D - gaussianKernal_2D_1
    bandpass = (bandpass-bandpass.min())/(bandpass.max()-bandpass.min())
    return bandpass

def positionKernel(shape,kernel,x,y):
    kernel_egde_mid = int(kernel.shape[0]/2)
    bandpass = kernel[kernel_egde_mid-y:kernel_egde_mid+shape[0]-y,kernel_egde_mid-x:kernel_egde_mid+shape[1]-x]
    bandpass = bandpass/np.sum(bandpass)
    return bandpass

def calcProbabilityMap(distance_map_pdf, position_kernel):
    prob_map = np.multiply(distance_map_pdf, position_kernel)
    prob_map = prob_map/np.sum(prob_map)
    return prob_map

def calcCDF(probability_map):
    pdf_1d = probability_map.ravel()
    cdf_1d = np.cumsum(pdf_1d)
    value, indices = np.unique(cdf_1d, return_index=True)
    value = value[1:]
    indices = indices[1:]
    return value,indices

def cdfFindNearest(array_index,array_cdf, rand_value):
    array_cdf = np.asarray(array_cdf)
    idx = (np.abs(array_cdf - rand_value)).argmin()
    return array_index[idx]

def calcPoint(rand_index, shape):
    x = np.mod(rand_index,shape[1])
    y = int(rand_index/shape[1])
    return x,y

def getRandomPoint(probability_map):
    rand_value=np.random.random_sample()
    value,indices = calcCDF(probability_map)
    index = cdfFindNearest(indices,value, rand_value)
    x,y = calcPoint(index, probability_map.shape)
    return x,y

def isAngleHold(previous_x,previous_y,previous_angle,max_angle,step_size,prob_map):
    new_x,new_y = getRandomPoint(prob_map)
    new_angle = np.arctan2((-new_y+previous_y),(new_x-previous_x))
    angle_dif = new_angle-previous_angle
    if angle_dif > np.pi:
        angle_dif = angle_dif - 2*np.pi
    if (abs(angle_dif)<=max_angle):
        x_step = previous_x + step_size*np.sin(new_angle)
        y_step = previous_y - step_size*np.cos(new_angle)
        return False,x_step,y_step,new_angle
    else:
        return True,None,None,None

initial_x = 100
initial_y = 435
initial_angle =80*np.pi/180
max_angle = 45*np.pi/180
step_size = 10 
pos = np.array([initial_y,initial_x])
pos = pos.reshape((1,2)).astype(np.float64)


kernel = np.ones((3,3), np.uint8) 
image = cv2.imread("D:\\FordOtosan\\occupancy.PNG",0)
#image = cv2.imread("D:\\_ravel.PNG",0)
#threshold image to get binary image
_,source_image = cv2.threshold(image,120,255,cv2.THRESH_BINARY)
img_erosion = cv2.erode(source_image, kernel, iterations=1) 
dist = cv2.distanceTransform(img_erosion, cv2.DIST_L2, 5)
dist = np.power(dist,2)
pdf = dist / np.sum(dist)
aas = 255*(dist-dist.min())/(dist.max()-dist.min())
cv2.imshow('eroded21', aas.astype(np.uint8))
gaussian_kernel = initGaussianKernel(dist.shape,7,2)
out = cv2.imread("D:\\FordOtosan\\occupancy.PNG",1)
while True:
    y,x = pos[pos.shape[0]-1,:]
    position_kernel = positionKernel(dist.shape,gaussian_kernel,int(x),int(y))
    prob_map = calcProbabilityMap(pdf, position_kernel)  
    img = 255*prob_map/prob_map.max()
    stop_cond = True
    while stop_cond:
        stop_cond,x_step,y_step,new_angle = isAngleHold(x,y,initial_angle,max_angle,step_size,prob_map)
        if stop_cond == False:
            if dist[int(y_step),int(x_step)] == 0:
                pos = pos[:pos.shape[0]-3,:]
                stop_cond = True
    new_point = np.array([y_step,x_step])
    new_point = new_point.reshape((1,2)).astype(np.float64)
    initial_angle = new_angle
    pos = np.vstack([pos, new_point])
    print_point = pos.astype(np.uint32)
    out = cv2.circle(out, (int(x_step), int(y_step)), int(3),(255, 0, 255), 1)
    cv2.imshow('eroded2', out.astype(np.uint8))
    cv2.imshow('eroded1', img.astype(np.uint8))
    time.sleep(0.4)
    if cv2.waitKey(27) & 0xFF == ord('q') :
        break

cv2.destroyAllWindows()
