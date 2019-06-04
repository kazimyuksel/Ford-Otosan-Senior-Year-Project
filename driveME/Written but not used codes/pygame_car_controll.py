# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 11:49:51 2019

@author: Kazım
"""
import sys
import pygame
import serial
import time
import numpy as np

com = serial.Serial(port = 'COM5', baudrate = 9600, parity = serial.PARITY_NONE, stopbits = serial.STOPBITS_ONE, bytesize = serial.EIGHTBITS, timeout = 0.4)

background_colour = (255,255,0)
(width, height) = (300, 200)
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption('Tutorial 1')
screen.fill(background_colour)
pygame.display.flip()
running = True

steer_byte = 35
throttle_byte = 38


_LEFT = 36
_RIGHT = 167
_MIDDLE = 110

_STOP = 102
_FORWARD = 120
_BACKWARD = 80

steer_pwm = _MIDDLE
throttle_pwm = _STOP

pressed_left = False
pressed_right = False
pressed_up = False
pressed_down = False

loop_time = 50 #ms
while running: 
    for event in pygame.event.get():
        if event.type == pygame.QUIT: 
            running = False       
        elif event.type == pygame.KEYDOWN:          # check for key presses          
            if event.key == pygame.K_LEFT:        # left arrow turns left
                pressed_left = True
            elif event.key == pygame.K_RIGHT:     # right arrow turns right
                pressed_right = True
            elif event.key == pygame.K_UP:        # up arrow goes up
                pressed_up = True
            elif event.key == pygame.K_DOWN:     # down arrow goes down
                pressed_down = True
        elif event.type == pygame.KEYUP:            # check for key releases
            if event.key == pygame.K_LEFT:        # left arrow turns left
                pressed_left = False
            elif event.key == pygame.K_RIGHT:     # right arrow turns right
                pressed_right = False
            elif event.key == pygame.K_UP:        # up arrow goes up
                pressed_up = False
            elif event.key == pygame.K_DOWN:     # down arrow goes down
                pressed_down = False
    
    # In your game loop, check for key states:
    steer_pwm = _MIDDLE
    throttle_pwm = _STOP
    if pressed_left:
        steer_pwm = _LEFT
    if pressed_right:
        steer_pwm = _RIGHT
    if pressed_right and pressed_left is True:
        steer_pwm = _MIDDLE
        
    if pressed_up:
        throttle_pwm = _FORWARD
    if pressed_down:
        throttle_pwm = _BACKWARD
    if pressed_up and pressed_down is True:
        throttle_pwm = _STOP
    
    current_time = time.time()
    while ((time.time()-current_time)*1000) < loop_time:
        continue
    com.write(bytes([steer_byte]))
    com.write(bytes([steer_pwm]))
    com.write(bytes([throttle_byte]))
    com.write(bytes([throttle_pwm]))
    hash_function = 255 + steer_byte * steer_pwm - throttle_pwm
    hash_function = np.mod(hash_function, 255)
    hash_read = com.read(size=1)
    print("HASH: "+ str(hash_function), end=' ')
    print("HASH READ: "+ str((int.from_bytes(hash_read, "big"))))
    if (int)(hash_function) == int.from_bytes(hash_read, "big"):
        print("Hash is correct!" )
    elif len(hash_read) == 0:
        print("TIMEOUT")
    else:
        print("Wrong hash!")
    #print("Left:"+str(pressed_left)+"  Right: "+str(pressed_right)+"  Forward: "+str(pressed_up)+"  Backward: "+str(pressed_down))
    
    
    
    steer_pwm = _MIDDLE
    throttle_pwm = _STOP
    pygame.display.update()
    
steer_pwm = _MIDDLE
throttle_pwm = _STOP
#written_byte =  bytearray([steer_byte, steer_pwm, throttle_byte, throttle_pwm])
#com.write(written_byte)
com.write(bytes([steer_byte]))
com.write(bytes([steer_pwm]))
com.write(bytes([throttle_byte]))
com.write(bytes([throttle_pwm]))
com.close()
pygame.quit()





ss