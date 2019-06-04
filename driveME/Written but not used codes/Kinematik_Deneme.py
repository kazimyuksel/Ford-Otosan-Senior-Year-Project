# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 14:29:22 2019

@author: Kazım
"""
import numpy as np

trailer_track_width = 56.1
head_track_width = 27.0
head_wheel_base = 13.395
fifth_wheel_distance = 5.25

head_speed = 500 #pix/s
head_angle = 0
trailer_angle = 120*np.pi/180 
head_position = (100,100)
trailer_position = None
steer_angle = 0*np.pi/180
trailer_rear_x = head_position[0] + fifth_wheel_distance*np.cos(head_angle) + trailer_track_width*np.cos(trailer_angle+np.pi)
trailer_rear_y = head_position[1] + fifth_wheel_distance*np.sin(head_angle) + trailer_track_width*np.sin(trailer_angle+np.pi)
trailer_pos = (trailer_rear_x, trailer_rear_y)

def _kinematicSolver(head_speed, head_angle, trailer_angle, steer_angle, head_position, dt):

    lh = head_track_width
    lc = fifth_wheel_distance
    lt = trailer_track_width
    
    head_speed_x = head_speed*np.cos(head_angle)
    head_speed_y = head_speed*np.sin(head_angle)
    heading_dot = (head_speed/lh)*np.tan(steer_angle)
    head_new_pos_x = head_position[0] + dt*head_speed_x
    head_new_pos_y = head_position[1] + dt*head_speed_y
    head_angle_new = head_angle + dt*heading_dot
    
    theta_s = steer_angle
    theta_h = head_angle_new
    theta_t = trailer_angle
    theta_n = theta_t
    Vh = head_speed
    F1 = (Vh/lt)*(np.tan(theta_s)*np.cos(theta_h-theta_t)*lc/lh+np.sin(theta_h-theta_t))
    theta_t = theta_n + 0.25*dt*F1
    F2 = (Vh/lt)*(np.tan(theta_s)*np.cos(theta_h-theta_t)*lc/lh+np.sin(theta_h-theta_t))
    theta_t = theta_n +(3/32)*dt*F1 + (9/32)*dt*F2
    F3 = (Vh/lt)*(np.tan(theta_s)*np.cos(theta_h-theta_t)*lc/lh+np.sin(theta_h-theta_t))
    theta_t = theta_n + (1932/2197)*dt*F1 - (7200/2197)*dt*F2 + (7296/2197)*dt*F3
    F4 = (Vh/lt)*(np.tan(theta_s)*np.cos(theta_h-theta_t)*lc/lh+np.sin(theta_h-theta_t))
    theta_t = theta_n + (439/216)*dt*F1 - 8*dt*F2 + (3680/513)*dt*F3 - (845/4104)*dt*F4
    F5 = (Vh/lt)*(np.tan(theta_s)*np.cos(theta_h-theta_t)*lc/lh+np.sin(theta_h-theta_t))
    theta_t = theta_n - (8/27)*dt*F1 + 2*dt*F2 - (3544/2565)*dt*F3 + (1859/4104)*dt*F4  - (11/40)*dt*F5
    F6 = (Vh/lt)*(np.tan(theta_s)*np.cos(theta_h-theta_t)*lc/lh+np.sin(theta_h-theta_t))
    
    theta_n1 = theta_n + dt*((16/135)*F1+(6656/12825)*F3+(28561/56430)*F4-(9/50)*F5+(2/55)*F6)
    new_theta_t = theta_n1;
    trailer_rear_x = head_position[0] + lc*np.cos(theta_h) + lt*np.cos(theta_n1+np.pi)
    trailer_rear_y = head_position[1] + lc*np.sin(theta_h) + lt*np.sin(theta_n1+np.pi)
    trailer_new_pos = (trailer_rear_x, trailer_rear_y)
    head_new_pos = (head_new_pos_x, head_new_pos_y)
    return head_angle_new, new_theta_t, head_new_pos, trailer_new_pos
    
def forwardPredictor(delay, dt, trailer_pos, trailer_angle, head_pos, head_angle, steer_angle, head_speed):
    loop_count = delay/dt
    loop_count = np.round(loop_count)
    new_dt = delay/loop_count
    new_dt = new_dt/1000
    for i in range(int(loop_count)):
        head_angle, trailer_angle, head_pos, trailer_pos = _kinematicSolver(head_speed, head_angle, trailer_angle, steer_angle, head_position, new_dt)
    return head_angle, trailer_angle, head_pos, trailer_pos

delay = 300
dt = 10
head_pos = (0,0)
head_angle, trailer_angle, head_pos, trailer_pos = forwardPredictor(delay, dt, trailer_pos, trailer_angle, head_pos, head_angle, steer_angle, head_speed)
