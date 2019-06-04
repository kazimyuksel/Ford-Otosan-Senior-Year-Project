# -*- coding: utf-8 -*-
"""
Created on Sat May 11 12:32:06 2019

@author: Kazım
"""

from colorama import init, Fore, Style
init(convert=True)
import datetime
import argparse
from subprocess import Popen, PIPE
import subprocess
import redis
from redis import ConnectionError
import time
import os
import signal
import cv2
import numpy as np
from io import BytesIO
import pygame
import pickle

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--computerID", required=True, help="ID of the current computer.")
ap.add_argument("-t", "--connectionType", required=True, help="eth or wifi")
ap.add_argument("-c", "--cameraID", required=True, help="Camera ID to be used.")
args = vars(ap.parse_args())


def readAndModify_RedisConfFile(path,bind_ip):
    print()
    print(f'{Fore.BLUE}#####################################################{Style.RESET_ALL}')
    print(f'{Fore.GREEN}readAndModify_RedisConfFile(){Style.RESET_ALL} -> Function is being {Fore.GREEN}executed{Style.RESET_ALL}!')
    print('Start time of module execution: '+ str(datetime.datetime.now()))
    print()
    conf_path = path + "driveME_server.conf"
    print("Configuration File Path: "+conf_path)
    f=open(conf_path, "r")
    print("File opened for reading.")
    if f.mode == 'r':
        content =f.read()
        print("File content read.")
    f.close()
    print("File closed.")
    new_content = ""
    for line in content.split("\n"):
        if "bind" in line:
            line = "bind "+bind_ip
            print("Bind IP is replaced.")
        if line != "":
            new_content = new_content + line +"\n"
    del line, content, conf_path
    print("Returning Content.")
    print(f'{Fore.GREEN}#####################################################{Style.RESET_ALL}')
    print()
    return new_content

def write_RedisConfFile(path, content):
    print()
    print(f'{Fore.BLUE}#####################################################{Style.RESET_ALL}')
    print(f'{Fore.GREEN}write_RedisConfFile(){Style.RESET_ALL} -> Function is being {Fore.GREEN}executed{Style.RESET_ALL}!')
    print('Start time of module execution: '+ str(datetime.datetime.now()))
    print()
    conf_path = path + "driveME_server.conf"
    print("Configuration File Path: "+conf_path)
    f = open(conf_path, "w+")
    print("File opened for writing.")
    if f.mode == 'w+':
        f.write(content)
        print("File content replaced.")
    f.close()
    print("File closed.")
    print(f'{Fore.GREEN}#####################################################{Style.RESET_ALL}')
    print()
    del conf_path

def pingIP(IP):
    word1 = "Destination"
    word2 = "timed out"
    toping = Popen(["ping","-n","1", IP], stdout=PIPE)
    output = toping.communicate()[0].decode("utf-8")
    isAlive = not ((word1 in output) or (word2 in output))
    toping.kill()
    return isAlive

def pingIPforHost(IP):
    word = "Maximum = 0ms"
    toping = Popen(["ping","-n","1", IP], stdout=PIPE)
    output = toping.communicate()[0].decode("utf-8")
    isHost = word in output
    toping.kill()
    return isHost


def runshellRedisServer(hostIP):
    print(f'{Fore.BLUE}#####################################################{Style.RESET_ALL}')
    print(f'{Fore.GREEN}runshellRedisServer(){Style.RESET_ALL} -> Function is being {Fore.GREEN}executed{Style.RESET_ALL}!')
    print('Start time of module execution: '+ str(datetime.datetime.now()))
    print()
    conf_content = readAndModify_RedisConfFile('C:/Program Files/Redis/',hostIP)
    print("Modified .conf file.")
    write_RedisConfFile('C:/Program Files/Redis/', conf_content)
    print("Saved .conf file.")
    print("Returned subprocess of redis-server.exe cmd.")
    print()
    print(f'{Fore.GREEN}#####################################################{Style.RESET_ALL}')
    print()
    return Popen('redis-server.exe driveME_server.conf', shell=True, cwd='C:/Program Files/Redis')

def killshellRedisServer(server):
    print(f'{Fore.BLUE}#####################################################{Style.RESET_ALL}')
    print(f'{Fore.GREEN}killshellRedisServer(){Style.RESET_ALL} -> Function is being {Fore.GREEN}executed{Style.RESET_ALL}!')
    print('Start time of module execution: '+ str(datetime.datetime.now()))
    print()
    os.kill(server.pid, signal.CTRL_C_EVENT)
    print("Ctrl+C event is sent to subprocess.")
    server.kill()
    print("Killed subprocess.")
    print()
    print(f'{Fore.GREEN}#####################################################{Style.RESET_ALL}')
    print()
    
def killRedisServerOverClient(hostIP):
    print(f'{Fore.BLUE}#####################################################{Style.RESET_ALL}')
    print(f'{Fore.GREEN}killRedisServerOverClient(){Style.RESET_ALL} -> Function is being {Fore.GREEN}executed{Style.RESET_ALL}!')
    print('Start time of module execution: '+ str(datetime.datetime.now()))
    print()
    command = "redis-cli -h "+hostIP+" -p 6379 shutdown"
    proc = Popen(command, shell=False, cwd='C:/Program Files/Redis')
    print("Redis Server is shutdown via Redis-Cli.")
    print()
    time.sleep(1)
    del proc
    print(f'{Fore.GREEN}#####################################################{Style.RESET_ALL}')
    print()
    
def pingRedisServer(hostIP):
    print(f'{Fore.BLUE}#####################################################{Style.RESET_ALL}')
    print(f'{Fore.GREEN}pingRedisServer(){Style.RESET_ALL} -> Function is being {Fore.GREEN}executed{Style.RESET_ALL}!')
    print('Start time of module execution: '+ str(datetime.datetime.now()))
    print()
    store = redis.Redis(host = hostIP, port = 6379)
    print("Redis server object created.")
    try:
        print("Server Pinged.")
        store.ping()
        print("Server up-and-running.")
        print()
        del store
        print(f'{Fore.GREEN}#####################################################{Style.RESET_ALL}')
        print()
        return True
    except (ConnectionError):
        print("No Server up-and-running.")
        print()
        del store
        print(f'{Fore.GREEN}#####################################################{Style.RESET_ALL}')
        print()
        return False
    
def testCamera(hostIP,CameraID,cameraIP):
    store = redis.Redis(host = hostIP, port = 6379)
    camera_ip = cameraIP[CameraID]
    camera_username = "admin"
    camera_password = "driveME2019"
    camera_address = "rtsp://"+camera_username+":"+camera_password+"@"+camera_ip+"/Streaming/Channels/2/picture"
    max_sleep = 5.0
    cur_sleep = 0.1
    key = "test_img"
    key_id = "test_img_id"
    while True:
        cap = cv2.VideoCapture(camera_address)
        if cap.isOpened():
            break
        print('not opened, sleeping {}s'.format(cur_sleep))
        time.sleep(cur_sleep)
        if cur_sleep < max_sleep:
            cur_sleep *= 2
            cur_sleep = min(cur_sleep, max_sleep)
            continue
        cur_sleep = 0.1
    counter = 0
    while True:
        _, image = cap.read()
        if image is None:
            time.sleep(0.05)
            continue
        _, image = cv2.imencode('.png', image)
        sio = BytesIO()
        np.save(sio, image)
        value = sio.getvalue()
        store.set(key, value)
        image_id = os.urandom(8)
        store.set(key_id, image_id)
        counter = counter+1
        if counter == 100:
            break
    if image is not None:
        image_get = store.get(key)
        image_get = BytesIO(image_get)
        image_get = np.load(image_get)
        rgb = cv2.imdecode(image_get, -1)
        if rgb is not None:
            return True, rgb
    return False, None

DIM = (640,480)
def initPyGame(DIM, caption):
    pygame.init()
    pygame.display.set_caption(caption)
    screen = pygame.display.set_mode([DIM[0],DIM[1]])
    return screen

def closePyGame():
    pygame.quit()
    
def displayImage(image,screen):
    screen.fill([0,0,0])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.rot90(image)
    image = cv2.flip( image, 0 )
    image = pygame.surfarray.make_surface(image)
    screen.blit(image, (0,0))
    pygame.display.update()
    
def checkMouseClick():
    for event in pygame.event.get():
        if event.type == pygame.MOUSEBUTTONDOWN:
            x, y = pygame.mouse.get_pos()
            return True, x, y
    return False, None, None
def checkExit():
    for event in pygame.event.get():
        if event.type == pygame.QUIT: 
            return False 
    return True
def calibrateCamera(img_arr,camera_id,index,surface):
    
    cam_id = camera_id[index]
    img_master = img_arr[index]
    img_master = undistortImage(img_master)
    caption = surface + " level calibration, cam: "+str(cam_id)
    screen = initPyGame(DIM, caption)
    
    running = True
    counter = 0
    major_counter = 0
    corners_array_master = np.empty((0,4,2))
    corner_array = np.empty((0,2))
    while running:
        ret = False
        img = img_master.copy()
        for point in corner_array:
            point = point.astype(np.int32)
            #point[0] = 640- point[0]
            img = cv2.circle(img,tuple(point),2,(0,255,0))
        displayImage(img,screen)
        ret, x, y = checkMouseClick()
        running = checkExit()
        if ret:
            counter = counter + 1
            corner = [x, y]
            corner = np.array(corner, dtype = np.float64)
            corner = corner.reshape((1,2))
            corner_array = np.concatenate((corner_array,corner))
        if counter == 4:
            corner_array = corner_array.reshape((1,4,2))
            corners_array_master = np.concatenate((corners_array_master,corner_array))
            corner_array = np.empty((0,2))
            counter = 0
            major_counter = major_counter + 1
        if major_counter == 5:
            running = False
    closePyGame()
    cv2.destroyAllWindows()
    
    pts_src = corners_array_master.mean(axis=0)
    #pts_s = pts_src.reshape((4,1,2))
    
    bottom_left = input("Enter bottom left corner coordinates (as x,y):  ")
    bottom_left_x, bottom_left_y = bottom_left.split(',')
    bottom_left_x, bottom_left_y = int(bottom_left_x), int(bottom_left_y)
    top_right = input("Enter top right corner coordinates (as x,y):  ")
    top_right_x, top_right_y = top_right.split(',')
    top_right_x, top_right_y = int(top_right_x), int(top_right_y)
    port_shape = (492,492) #in CM scale. 1pixel = 1cm
    port_tile_count = (9,9) 
    port_tile_size = ((port_shape[0]/port_tile_count[0]),(port_shape[1]/port_tile_count[1]))
    corner_1 = np.array([bottom_left_x*port_tile_size[0],np.float64(port_shape[1]) - bottom_left_y*port_tile_size[1]],dtype=np.float64).reshape((1,2))
    corner_2 = np.array([top_right_x*port_tile_size[0],np.float64(port_shape[1]) - bottom_left_y*port_tile_size[1]],dtype=np.float64).reshape((1,2))
    corner_3 = np.array([top_right_x*port_tile_size[0],np.float64(port_shape[1]) - top_right_y*port_tile_size[1]],dtype=np.float64).reshape((1,2))
    corner_4 = np.array([bottom_left_x*port_tile_size[0],np.float64(port_shape[1]) - top_right_y*port_tile_size[1]],dtype=np.float64).reshape((1,2))
    
    pts_dst = np.empty((0,2))
    pts_dst = np.concatenate((pts_dst,corner_1))
    pts_dst = np.concatenate((pts_dst,corner_2))
    pts_dst = np.concatenate((pts_dst,corner_3))
    pts_dst = np.concatenate((pts_dst,corner_4))
    h, status = cv2.findHomography( pts_src, pts_dst )
    img = img_master
    out = cv2.warpPerspective( img, h, port_shape ) 
    cv2.imshow("warped",out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    roi_top = input("ROI top coordinate (as y):  ")
    roi_top = int(int(roi_top)*np.float64(port_tile_size[1]))
    if surface == "ground":
        pickle_out = open("camera_calibration_ground_"+str(cam_id)+".driveme","wb")
    else:
        pickle_out = open("camera_calibration_trailer_"+str(cam_id)+".driveme","wb")
    pickle.dump(roi_top, pickle_out)
    pickle.dump(h, pickle_out)
    pickle_out.close()
    
def createRelativeTransform(camera_id,index):
    cam_id = camera_id[index]
    f = open("camera_calibration_ground_"+str(cam_id)+".driveme", "rb")
    roi_ground = pickle.load(f)
    h_ground = pickle.load(f)
    f.close()
    time.sleep(1)
    f = open("camera_calibration_trailer_"+str(cam_id)+".driveme", "rb")
    roi_trailer = pickle.load(f)
    h_trailer = pickle.load(f)
    f.close()
    del roi_ground, roi_trailer
    h_ground_inv =  np.linalg.inv(h_ground)
    h_relative = np.matmul(h_trailer,h_ground_inv)
    pickle_out = open("camera_calibration_relative_"+str(cam_id)+".driveme","wb")
    pickle.dump(h_relative, pickle_out)
    pickle_out.close()
    
def undistortImage(img):
    DIM = (640,480)
    K=np.array([[344.8127130210307, 0.0, 336.40818490780066], [0.0, 464.63383194201896, 241.32850767967614], [0.0, 0.0, 1.0]])
    D=np.array([[0.056114758269676296], [-0.2394852281183658], [0.3620424153302755], [-0.18229098502853122]])
    
    dim1 = DIM
    dim2 = (640, 480)
    dim3 = (640, 480)
    balance=1.0
    scaled_K = K * dim1[0] / DIM[0]  # The values of K is to scale with image dimension.
    scaled_K[2][2] = 1.0  # Except that K[2][2] is always 1.0
    # This is how scaled_K, dim2 and balance are used to determine the final K used to un-distort image. OpenCV document failed to make this clear!
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, D, dim2, np.eye(3), balance=balance)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, D, np.eye(3), new_K, dim3, cv2.CV_16SC2)
    return cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT) 
        
    
    
    
    
    
###############################################################################
#MAIN STARTS HERE
    
try:
    hostID = args["computerID"]
    connectionType = args["connectionType"]
    if connectionType == "eth":
        hostIP = "192.168.1.10"+hostID
    else:
        hostIP = "192.168.1.20"+hostID
        
    camera_id_str = args["cameraID"]
    camera_id = args["cameraID"].split(",")
    
    print(f'{Fore.BLUE}#####################################################{Style.RESET_ALL}')
    print(f'{Fore.GREEN}slave_main.py{Style.RESET_ALL} -> Code is being {Fore.GREEN}executed{Style.RESET_ALL}!')
    print('Start time of code execution: '+ str(datetime.datetime.now()))
    print('Welcome!')
    print()
    print("Host ID:\t"+str(hostID))
    print("Host Connection Type:\t"+str(connectionType))
    print("Host IP:\t"+str(hostIP))
    print("Camera IDs:\t"+str(camera_id_str))
    print(f'{Fore.GREEN}#####################################################{Style.RESET_ALL}')
    print()
    
    #CONSTANTS
    isServerUp = False
    redis_configuration_path = "D:/"
    
    camera_username = "admin"
    camera_password = "driveME2019"
    
    gatewayIP = "192.168.1.1"
    cameraIP = [None]
    cameraIP.append("192.168.1.11")
    cameraIP.append("192.168.1.12")
    cameraIP.append("192.168.1.13")
    cameraIP.append("192.168.1.14")
    
    isServerUp = pingRedisServer(hostIP)
    time.sleep(1)
    if not isServerUp:
        shell = runshellRedisServer(hostIP)
        time.sleep(1)
        isServerUp = pingRedisServer(hostIP)
    
    cam_connected = True
    for cam_id in camera_id:
        ping_status = pingIP(cameraIP[int(cam_id)])
        if ping_status == False:
            cam_connected = False
        print("Camera "+str(cam_id)+" is working?: "+str(ping_status))
    print()
    cam_alive = True
    img_arr = []
    if cam_connected == True and isServerUp == True:
        for cam_id in camera_id:
            cam_id = int(cam_id)
            cam_status, img = testCamera(hostIP,cam_id,cameraIP)
            img_arr.append(img)
            if cam_status == False:
                cam_alive = False
            print("Camera "+str(cam_id)+" is getting image and working with server?: "+str(cam_status))
    print()
    
    
    isCalibrationExist = True
    wantCalibrate = False
    for cam in camera_id:
        try:
            fh_1 = open("camera_calibration_ground_"+str(cam)+".driveme", "r")
            fh_2 = open("camera_calibration_trailer_"+str(cam)+".driveme", "r")
            fh_1.close()
            fh_2.close()
        except FileNotFoundError:
            isCalibrationExist = False
            print("Calibration file does not exist")
            print("Starting Calibration")
            for i,img in enumerate(img_arr):
                calibrateCamera(img_arr,camera_id,i,surface="ground")
                
            cam_alive = True
            _ = input("Enter any thing, if you lowered cameras to ground level?:  ")
            img_arr = []
            if cam_connected == True and isServerUp == True:
                for cam_id in camera_id:
                    cam_id = int(cam_id)
                    cam_status, img = testCamera(hostIP,cam_id,cameraIP)
                    img_arr.append(img)
                    if cam_status == False:
                        cam_alive = False
                    print("Camera "+str(cam_id)+" is getting image and working with server?: "+str(cam_status))
            print()
            for i,img in enumerate(img_arr):
                calibrateCamera(img_arr,camera_id,i,surface="trailer")
            for i,img in enumerate(img_arr):
                createRelativeTransform(camera_id,i)
    if isCalibrationExist:
        wantCalibrate = input("Do you want to calibrate?:  ")=="y"
        if wantCalibrate:
            print("Starting Calibration")
            for i,img in enumerate(img_arr):
                calibrateCamera(img_arr,camera_id,i,surface="ground")
                
            cam_alive = True
            _ = input("Enter any thing, if you lowered cameras to ground level?:  ")
            img_arr = []
            if cam_connected == True and isServerUp == True:
                for cam_id in camera_id:
                    cam_id = int(cam_id)
                    cam_status, img = testCamera(hostIP,cam_id,cameraIP)
                    img_arr.append(img)
                    if cam_status == False:
                        cam_alive = False
                    print("Camera "+str(cam_id)+" is getting image and working with server?: "+str(cam_status))
            print()
            for i,img in enumerate(img_arr):
                calibrateCamera(img_arr,camera_id,i,surface="trailer")
            for i,img in enumerate(img_arr):
                createRelativeTransform(camera_id,i)
            isCalibrationExist = True   
   
    
    if isServerUp:
        killRedisServerOverClient(hostIP)
        time.sleep(1)
        isServerUp = pingRedisServer(hostIP)
        
    shell = runshellRedisServer(hostIP)
    time.sleep(1)
    isServerUp = pingRedisServer(hostIP)
    time.sleep(1)
    if cam_connected == True and isServerUp == True:
        recorder_process = []
        full_path = os.path.realpath(__file__)
        path, filename = os.path.split(full_path)
        camera_username = "admin"
        camera_password = "driveME2019"
        path = path+"\\"
        print()
        print("Executing Buffer cleaning shells.")
        for cam_id in camera_id:
            camera_ip = cameraIP[int(cam_id)]
            recorder_command = "python ipcam_recorder.py "
            recorder_command = recorder_command + "--cameraid "+str(cam_id)+" "
            recorder_command = recorder_command + "--username "+camera_username+" "
            recorder_command = recorder_command + "--password "+camera_password+" "
            recorder_command = recorder_command + "--cameraip "+str(camera_ip)+" "
            recorder_command = recorder_command + "--serverip "+str(hostIP)
            print("Path of the command to be executed: "+path)
            print("Shell command that will be executed: "+recorder_command)
            recorder_process.append(subprocess.Popen(recorder_command, shell=True, cwd=path)) 
            print()
        time.sleep(2)
        ticker_command = "python fps_monitor.py "
        ticker_command = ticker_command + "--cameraID " + camera_id_str + " "
        ticker_command = ticker_command + "--serverIP " + str(hostIP)
        fps_ticker = subprocess.Popen(ticker_command, shell=True, cwd=path)
        print()
    else:
        pass
        #raiseError
        
except (KeyboardInterrupt):
    print("Keyboard Interrupt.")
    try:
        shell
        killRedisServerOverClient(hostIP)
    except (NameError):
        pass
    
    try:
        recorder_command
        for cmd in recorder_command:
            os.kill(cmd.pid, signal.CTRL_C_EVENT)
    except (NameError):
        pass
    
    try:
        fps_ticker
        os.kill(fps_ticker.pid, signal.CTRL_C_EVENT)
    except (NameError):
        pass
            
        