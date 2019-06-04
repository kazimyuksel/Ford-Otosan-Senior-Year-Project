#def calculateSpeed(dt,orientation,prev_point,current_point):
#    acceptAngleThreshold = 25.0
#    prev_point_array = np.array(prev_point,dtype=np.float64).reshape((1,2))
#    current_point_array = np.array(current_point,dtype=np.float64).reshape((1,2))
#    del_position = current_point_array - prev_point_array
#    abs_speed = np.linalg.norm(del_position) / dt
#    print(abs_speed)
#    return abs_speed, current_point
#    if del_position[0,0] == 0.0 and del_position[0,1] == 0.0:
#        speed = 0.0
#        return speed, current_point
#    moving_direction = np.mod((np.arctan2(del_position[0,0],del_position[0,1])+2*np.pi),2*np.pi)
#    abs_speed = np.linalg.norm(del_position) / dt
#    orientation_error = orientation - moving_direction
#    if orientation_error > np.pi:
#        orientation_error = orientation_error - 2*np.pi
#    print(np.abs(orientation_error))
#    acceptAngleThreshold = acceptAngleThreshold*np.pi/180.0
#    if np.abs(orientation_error) <= acceptAngleThreshold:
#        speed = abs_speed
#    elif np.abs(orientation_error) >= (np.pi - acceptAngleThreshold) and np.abs(orientation_error) <= (np.pi + acceptAngleThreshold):
#        speed = -abs_speed
#    else:
#        speed = 0.0
#    
#    return speed, current_point
import argparse
import redis
import time
import numpy as np
from io import BytesIO
import os
import math
import cv2


eps  = math.ldexp(1.0, -53)
ap = argparse.ArgumentParser()
#1,eth,1,3,#,2,eth,2,4
ap.add_argument("-c", "--connectionList", required=True, help="Connection list with , and # delimited. ex: 1,eth,1,3#2,eth,2,4")
ap.add_argument("-i", "--currentHostID", required=True, help="Connection type of computer ID")
ap.add_argument("-t", "--currentHostConnectionType", required=True, help="Connection type of computer. wifi or eth")
args = vars(ap.parse_args())

prev_data_id = [None,None,None,None]
def getData():
    head_x = []
    head_y = []
    head_angle = []
    trailer_x = []
    trailer_y = []
    trailer_angle = []
    for index,server_obj in enumerate(slave_redis):
        for cam_id in connected_camera_list[index]:
            local_cam_data_key = cam_data_key + str(cam_id)
            local_cam_id_key = cam_id_key + str(cam_id)
            while True:
                time.sleep(0.01)
                position_data_id = server_obj.get(local_cam_id_key)
                if position_data_id != prev_data_id[int(cam_id)-1]:
                    break
            prev_data_id[int(cam_id)-1] = position_data_id
            data_get = server_obj.get(local_cam_data_key)
            data_get = BytesIO(data_get)
            raw_data = np.load(data_get,allow_pickle=True)
            
            head_x.append(raw_data[0])
            head_y.append(raw_data[1])
            head_angle.append(raw_data[2])
            trailer_x.append(raw_data[3])
            trailer_y.append(raw_data[4])
            trailer_angle.append(raw_data[5])
            
    head_x = np.nanmean(np.array(head_x,dtype=np.float64))
    head_y = np.nanmean(np.array(head_y,dtype=np.float64))
    head_angle = head_angle[np.logical_not(np.isnan(head_angle))]

    trailer_x = np.nanmean(np.array(trailer_x,dtype=np.float64))
    trailer_y = np.nanmean(np.array(trailer_y,dtype=np.float64))
    trailer_angle = trailer_angle[np.logical_not(np.isnan(trailer_angle))]
    
    x = 0
    y = 0
    for ang in head_angle:
        x = x+np.sin(ang)
        y = y+np.cos(ang)
    x = x / head_angle.shape[0]
    y = y / head_angle.shape[0]
    head_angle = np.arctan2(y,x)
    
    x = 0
    y = 0
    for ang in trailer_angle:
        x = x+np.sin(ang)
        y = y+np.cos(ang)
    x = x / trailer_angle.shape[0]
    y = y / trailer_angle.shape[0]
    trailer_angle = np.arctan2(y,x)
    
    
    relative_angle = head_angle - trailer_angle
    if relative_angle > np.pi:
        relative_angle = relative_angle - 2*np.pi
        
    return (head_x,head_y),(trailer_x,trailer_y),head_angle,trailer_angle,relative_angle

def calculateSpeed(prev_speed,prev_point,current_point,head_orientation,dt):
    if np.isnan(current_point[0]) or np.isnan(head_orientation):
        if np.isnan(current_point[0]) and not np.isnan(prev_point[0]):
            return prev_speed, prev_point
        elif not np.isnan(current_point[0]) and np.isnan(prev_point[0]):
            return prev_speed, current_point
    delta_position = np.array([current_point[0]-prev_point[0],current_point[1]-prev_point[1]],dtype =np.float64)
    abs_speed = np.linalg.norm(np.array([delta_position])) / (dt+eps)
    direction_unit_vector = np.array([np.cos(head_orientation),np.sin(head_orientation)],dtype=np.float64)
    direction_dot = np.dot(direction_unit_vector,delta_position)
    if direction_dot >= 0.0:
        current_speed = abs_speed
    else:
        current_speed = abs_speed
    if prev_speed is not None:
        speed = 0.8*prev_speed + 0.2*current_speed
        return speed, current_point
    else:
        return current_speed, current_point

def calculateYawRate(prev_yaw,prev_angle,current_angle,dt):
    if np.isnan(current_angle):
        return prev_yaw, prev_angle
    current_yaw = current_angle - prev_angle
    if current_yaw > np.pi:
        current_yaw = current_yaw - 2*np.pi
    current_yaw = current_yaw / (dt+eps)
    if prev_yaw is not None:
        yaw_rate = 0.8*prev_yaw + 0.2*current_yaw
        return yaw_rate,current_angle
    else:
        return current_yaw,current_angle
connectionList=[]
connectionList = args["connectionList"]
#connectionList = "1,eth,1,2,3,4"
connectionList = connectionList.split("#")
connectionList_master = []
for item in connectionList:
    delimited_item = item.split(",")
    connectionList_master.append(delimited_item)
    
hostID = args["currentHostID"]
hostCON = args["currentHostConnectionType"]
if hostCON == "eth":
    hostIP = "192.168.1.10"+str(hostID)
else:
    hostIP = "192.168.1.20"+str(hostID)

slave_redis = []
slave_ip_list = []
computer_id_list = []
connected_camera_list = []
for item_master in connectionList_master:
    item = item_master.copy()
    computer_id = item.pop(0)
    computer_id_list.append(computer_id)
    slave_con = item.pop(0)
    if slave_con == "eth":
        slave_ip = "192.168.1.10"+str(computer_id)
    else:
        slave_ip = "192.168.1.20"+str(computer_id)
    slave_ip_list.append(slave_ip)
    slave_redis.append(redis.Redis(host=slave_ip,port=6379))
    cam_id_list = []
    for cam_id in item:
        cam_id_list.append(cam_id)
    connected_camera_list.append(cam_id_list)


cam_data_key = "cam_data_"
cam_id_key = "cam_data_id_"
dt = 0.05
elapsed_time = 0.05

host_redis = redis.Redis(host=hostIP,port=6379)
headPos,trailerPos,headAngle,trailerAngle,relativeAngle = getData()
prev_point = headPos
prev_angle = trailerAngle
headSpeed, headYawRate = 0.0,0.0
headSpeed,prev_point=calculateSpeed(headSpeed,prev_point,headPos,headAngle,dt)
headYawRate,prev_angle = calculateYawRate(headYawRate,prev_angle,trailerAngle,dt)
start_time = time.time()

headSpeed = 0.0
headYawRate = 0.0

kf_head = cv2.KalmanFilter(4, 2, 0, cv2.CV_32F)
kf_head.transitionMatrix = np.array([[1.,0.,0.05,0.],[0.,1.,0.,0.05],[0.,0.,1.,0.],[0.,0.,0.,1.]])
kf_head.measurementMatrix = np.array([[1.,0.,0.,0.],[0.,1.,0.,0.]])
kf_head.processNoiseCov = 1e-2 * np.eye(4)
kf_head.measurementNoiseCov = 0.01 * np.eye(2)
kf_head.errorCovPost = 1. * np.eye(4)
kf_head.statePost = 1. * np.array([[headPos[0]],[headPos[1]],[0.],[0.]])  

kf_trailer = cv2.KalmanFilter(4, 2, 0, cv2.CV_32F)
kf_trailer.transitionMatrix = np.array([[1.,0.,0.05,0.],[0.,1.,0.,0.05],[0.,0.,1.,0.],[0.,0.,0.,1.]])
kf_trailer.measurementMatrix = np.array([[1.,0.,0.,0.],[0.,1.,0.,0.]])
kf_trailer.processNoiseCov = 1e-2 * np.eye(4)
kf_trailer.measurementNoiseCov = 0.01 * np.eye(2)
kf_trailer.errorCovPost = 1. * np.eye(4)
kf_trailer.statePost = 1. * np.array([[trailerPos[0]],[trailerPos[1]],[0.],[0.]])  
 
kf_trailer_angle = cv2.KalmanFilter(2, 1, 0, cv2.CV_32F)
kf_trailer_angle.transitionMatrix = np.array([[1., 1.], [0., 1.]])
kf_trailer_angle.measurementMatrix = 1. * np.ones((1, 2))
kf_trailer_angle.processNoiseCov = 1e-2 * np.eye(2)
kf_trailer_angle.measurementNoiseCov = 1e-2 * np.ones((1, 1))
kf_trailer_angle.errorCovPost = 1. * np.ones((2, 2))
kf_trailer_angle.statePost = np.array([[trailerAngle],[0.]])  

kf_head_angle = cv2.KalmanFilter(2, 1, 0, cv2.CV_32F)
kf_head_angle.transitionMatrix = np.array([[1., 1.], [0., 1.]])
kf_head_angle.measurementMatrix = 1. * np.ones((1, 2))
kf_head_angle.processNoiseCov = 1e-2 * np.eye(2)
kf_head_angle.measurementNoiseCov = 1e-2 * np.ones((1, 1))
kf_head_angle.errorCovPost = 1. * np.ones((2, 2))
kf_head_angle.statePost = np.array([[trailerAngle],[0.]])  

    

loop_frequency = 20.0 # Hz

headSpeed_lpf = 0.0
headYawRate_lpf = 0.0
prev_head_pos = np.array([headPos[0],headPos[1]])
prev_head_yawrate = headAngle
prev_trailer_angle = 0
prev_head_angle = 0
while True:
    loop_time = time.time()
    dt = 0.9*dt + 0.1*elapsed_time
    headPos,trailerPos,headAngle,trailerAngle,relativeAngle = getData()

    kf_head.transitionMatrix = np.array([[1.,0.,dt,0.],[0.,1.,0.,dt],[0.,0.,1.,0.],[0.,0.,0.,1.]])
    kf_trailer.transitionMatrix = np.array([[1.,0.,dt,0.],[0.,1.,0.,dt],[0.,0.,1.,0.],[0.,0.,0.,1.]])
    kf_trailer_angle.transitionMatrix = np.array([[1., dt], [0., dt]])
    measurement_head = 1. * np.array([[0.],[0.]])
    measurement_head[0] = 1. * headPos[0]
    measurement_head[1] = 1. * headPos[1]
    prediction = kf_head.predict()
    if not np.isnan(np.float64(measurement_head[0])):
        kf_head.correct(measurement_head)
    headPos = (kf_head.statePost[0],kf_head.statePost[1])
    
    measurement_trailer = 1. * np.array([[0.],[0.]])
    measurement_trailer[0] = 1. * trailerPos[0]
    measurement_trailer[1] = 1. * trailerPos[1]
    
    prediction = kf_trailer.predict()
    if not np.isnan(np.float64(measurement_trailer[0])):
        kf_trailer.correct(measurement_trailer)
    trailerPos = (kf_trailer.statePost[0],kf_trailer.statePost[1])
    
    measurement_trailer_angle = 1. * np.array([[0.]])
    measurement_trailer_angle[0] = 1. * trailerAngle
    
    prediction = kf_trailer_angle.predict()
    if not np.isnan(np.float64(measurement_trailer_angle[0])):
        trailerAngle = measurement_trailer_angle
        prev_trailer_angle = measurement_trailer_angle
    else:
        trailerAngle = prev_trailer_angle
#        if measurement_trailer_angle[0]>np.pi:
#            measurement_trailer_angle[0] = measurement_trailer_angle[0] -2*np.pi
#        elif measurement_trailer_angle[0]<np.pi:
#            measurement_trailer_angle[0] = measurement_trailer_angle[0] +2*np.pi
#        kf_trailer_angle.correct(measurement_trailer_angle)
#    if kf_trailer_angle.statePost[0,0]>np.pi:
#        kf_trailer_angle.statePost[0,0] = kf_trailer_angle.statePost[0,0] -2*np.pi
#    elif kf_trailer_angle.statePost[0,0]<np.pi:
#        kf_trailer_angle.statePost[0,0] = kf_trailer_angle.statePost[0,0] +2*np.pi
#    trailerAngle = kf_trailer_angle.statePost[0,0]
    
    measurement_head_angle = 1. * np.array([[0.]])
    measurement_head_angle[0] = 1. * headAngle
    
#    prediction = kf_head_angle.predict()
    if not np.isnan(np.float64(measurement_head_angle[0])):
        headAngle = measurement_head_angle
        prev_head_angle = measurement_head_angle
    else:
        headAngle = prev_head_angle
#        kf_head_angle.correct(measurement_head_angle)
#    headAngle = kf_head_angle.statePost[0,0]
    
#    
#    headSpeed_kf = np.sqrt(np.power(kf_head.statePost[2],2)+np.power(kf_head.statePost[3],2))
#    head_displacement=np.array([headPos[0],headPos[1]])-prev_head_pos
#    prev_head_pos = np.array([headPos[0],headPos[1]])
#    headSpeed = np.linalg.norm(head_displacement) / (dt)
#    direction_unit_vector = np.array([np.cos(headAngle),np.sin(headAngle)],dtype=np.float64)
#    direction_dot = np.dot(direction_unit_vector,np.array([kf_head.statePost[2],kf_head.statePost[2]]))
#    if direction_dot >= 0.0:
#        headSpeed = 0.7*headSpeed + 0.3*headSpeed_kf
#    else:
#        headSpeed = -(0.7*headSpeed + 0.3*headSpeed_kf)
##    headSpeed,prev_point=calculateSpeed(headSpeed,prev_point,headPos,headAngle,dt)
##    headYawRate,prev_angle = calculateYawRate(headYawRate,prev_angle,headAngle,dt)
#    headSpeed_lpf = 0.6*headSpeed_lpf + 0.4*headSpeed 
    
    del_yaw = headAngle - prev_angle
    prev_angle = headAngle
    if del_yaw > np.pi:
        del_yaw = del_yaw - 2*np.pi
    headYawRate = del_yaw / (dt)
    headYawRate_lpf = 0.6*headYawRate_lpf + 0.4*headYawRate
    
    
    if headSpeed_lpf >0.0:
        if headYawRate_lpf >0.0:
            output = "Code Hz: {:.2f}\t ,\t h_x: {:.2f}\t ,\t h_y: {:.2f}\t ,\t h_angle: {:.2f}\t ,\t t_x: {:.2f}\t ,\t t_y: {:.2f}\t ,\t t_angle: {:.2f}\t ,\t head_speed: {:.2f}\t ,\t head_yaw_rate: {:.2f}".format(np.float64(1.0/dt),np.float64(headPos[0]),np.float64(headPos[1]),np.float64(headAngle),np.float64(trailerPos[0]),np.float64(trailerPos[1]),np.float64(trailerAngle),np.float64(headSpeed_lpf),np.float64(headYawRate_lpf))
        else:
            output = "Code Hz: {:.2f}\t ,\t h_x: {:.2f}\t ,\t h_y: {:.2f}\t ,\t h_angle: {:.2f}\t ,\t t_x: {:.2f}\t ,\t t_y: {:.2f}\t ,\t t_angle: {:.2f}\t ,\t head_speed: {:.2f}\t ,\t head_yaw_rate:{:.2f}".format(np.float64(1.0/dt),np.float64(headPos[0]),np.float64(headPos[1]),np.float64(headAngle),np.float64(trailerPos[0]),np.float64(trailerPos[1]),np.float64(trailerAngle),np.float64(headSpeed_lpf),np.float64(headYawRate_lpf))
    else:
        if headYawRate_lpf >0.0:
            output = "Code Hz: {:.2f}\t ,\t h_x: {:.2f}\t ,\t h_y: {:.2f}\t ,\t h_angle: {:.2f}\t ,\t t_x: {:.2f}\t ,\t t_y: {:.2f}\t ,\t t_angle: {:.2f}\t ,\t head_speed:{:.2f}\t ,\t head_yaw_rate: {:.2f}".format(np.float64(1.0/dt),np.float64(headPos[0]),np.float64(headPos[1]),np.float64(headAngle),np.float64(trailerPos[0]),np.float64(trailerPos[1]),np.float64(trailerAngle),np.float64(headSpeed_lpf),np.float64(headYawRate_lpf))
        else:
            output = "Code Hz: {:.2f}\t ,\t h_x: {:.2f}\t ,\t h_y: {:.2f}\t ,\t h_angle: {:.2f}\t ,\t t_x: {:.2f}\t ,\t t_y: {:.2f}\t ,\t t_angle: {:.2f}\t ,\t head_speed:{:.2f}\t ,\t head_yaw_rate:{:.2f}".format(np.float64(1.0/dt),np.float64(headPos[0]),np.float64(headPos[1]),np.float64(headAngle),np.float64(trailerPos[0]),np.float64(trailerPos[1]),np.float64(trailerAngle),np.float64(headSpeed_lpf),np.float64(headYawRate_lpf))
    print(output)
    
    redis_data_write = np.append(headPos, headAngle)
    redis_data_write = np.append(redis_data_write, trailerPos)
    redis_data_write = np.append(redis_data_write, trailerAngle)
    redis_data_write = np.append(redis_data_write, relativeAngle)
    run_bool = True
    for i in redis_data_write:
        if i == None:
            run_bool = False
    if run_bool:
        sio = BytesIO() 
        np.save(sio, redis_data_write)
        value = sio.getvalue()
    
        host_redis.set("master_spatial_attributes_data", value)
        position_data_id = os.urandom(8)
        host_redis.set("master_spatial_attributes_id", position_data_id)
#    while (time.time()- loop_time) < 1.0/loop_frequency:
#        pass
    elapsed_time = time.time() - start_time
    start_time =time.time()