# -*- coding: utf-8 -*-
"""
Created on Wed May  1 12:08:44 2019

@author: Kazım
"""
from colorama import init, Fore, Style
init(convert=True)
import datetime
import argparse
import h5py
import sys
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



# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--testIP", required=True, help="y if you want to test PORTS, n if you want to import existing file.")
ap.add_argument("-s", "--saveIP", required=True, help="y if you want to save PARAMETERS, n if you want to make it single time operation.")
ap.add_argument("-w", "--acceptWifi", required=True, help="y if wifi connection is okay, n if eth is mandatory.")
ap.add_argument("-c", "--cameraID", required=True, help="Camera IP Adress")
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

def testConnection():
    print()
    print(f'{Fore.BLUE}#####################################################{Style.RESET_ALL}')
    print(f'{Fore.GREEN}testConnection(){Style.RESET_ALL} -> Function is being {Fore.GREEN}executed{Style.RESET_ALL}!')
    print('Start time of module execution: '+ str(datetime.datetime.now()))
    print()
    computerIP = []
    computerID = [1,2,3,4,5,6,7,8,9]
    computerCON = []
    tempID = []
    for i in computerID:
        isAlive = False
        conType = "eth"
        word1 = "Destination"
        word2 = "timed out"
        word3 = "Maximum = 0ms"
        ip = "192.168.1.10" + str(i)
        toping = Popen(["ping","-n","1", ip], stdout=PIPE)
        output = toping.communicate()[0].decode("utf-8")
        isAlive = (word1 in output) or (word2 in output)
        if (not isAlive) and word3 in output:
            hostIP = ip
        print("Is IP:"+ip+" dead?: "+str(isAlive))
        if isAlive == True:
            conType = "wifi"
            ip = "192.168.1.20" + str(i)
            toping = Popen(["ping","-n","1", ip], stdout=PIPE)
            output = toping.communicate()[0].decode("utf-8")
            isAlive = (word1 in output) or (word2 in output)
            if (not isAlive) and word3 in output:
                hostIP = ip
            print("Is IP:"+ip+" dead?: "+str(isAlive))
        if isAlive == False:
            tempID.append(i)
            computerIP.append(ip)
            computerCON.append(conType)
    if ("wifi" in computerCON) and (not wifi_accept):
        sys.exit("WiFi connection detected. User said it will not be allowed. EXITING.")
    computerID = tempID
    print()
    print("Following IPs are detected.")
    print(computerIP)
    print("Current computer's IP (hostIP) is "+hostIP)
    print()
    print(f'{Fore.GREEN}#####################################################{Style.RESET_ALL}')
    print()
    toping.kill()
    return computerIP, computerID, computerCON, hostIP

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
        if counter == 50:
            break
    if image is not None:
        image_get = store.get(key)
        image_get = BytesIO(image_get)
        image_get = np.load(image_get)
        rgb = cv2.imdecode(image_get, -1)
        if rgb is not None:
            return True
    return False
###############################################################################
#MAIN STARTS HERE
try:
    testIP = args["testIP"] == "y"
    saveIP_configuration = args["saveIP"] == "y"
    wifi_accept = args["acceptWifi"] == "y"
    camera_id_str = args["cameraID"]
    camera_id = args["cameraID"].split(",")
    
    print(f'{Fore.BLUE}#####################################################{Style.RESET_ALL}')
    print(f'{Fore.GREEN}slave_main.py{Style.RESET_ALL} -> Code is being {Fore.GREEN}executed{Style.RESET_ALL}!')
    print('Start time of code execution: '+ str(datetime.datetime.now()))
    print('Welcome!')
    print()
    print("Computer Ports and Camera Ports will be tested:\t"+str(testIP))
    print("Parameters file be saved to root directory:\t"+str(saveIP_configuration))
    print("WiFi connection is slow but will be accepted:\t"+str(wifi_accept))
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
    
    if testIP is False:
        # import parameters
        testIP = False
        try:
            with  h5py.File('COMPUTER_PARAMETERS.h5') as hf: 
                computerID, computerIP, computerCON = hf['ids'][:], hf['ips'][:], hf['con'][:]
            computerID = list(computerID)   
            computerIP = [n.decode("ascii", "ignore") for n in computerIP]
            computerCON = [n.decode("ascii", "ignore") for n in computerCON]
            print("Importing COMPUTER_PARAMETERS.h5 is successful.")
            print("Imported parameters:")
            print(computerID)
            print(computerIP)
            print(computerCON)
            print("IPs will NOT be tested.")
            print()
            if ("wifi" in computerCON) and (not wifi_accept):
                sys.exit("WiFi connection detected. User said it will not be allowed. EXITING.")
        except (IOError,OSError, KeyError):
            print("Importing COMPUTER_PARAMETERS.h5 failed.")
            print("IPs will be tested.")
            print()
            testIP = True
    
    if testIP is True:
        computerIP, computerID, computerCON, hostIP = testConnection()
        isServerUp = pingRedisServer(hostIP)
        time.sleep(1)
        if not isServerUp:
            shell = runshellRedisServer(hostIP)
            time.sleep(1)
            isServerUp = pingRedisServer(hostIP)
    if testIP is False:
        for test_ip in computerIP:
           host_status = pingIPforHost(test_ip)
           if host_status:
               hostIP = test_ip
               break
    if saveIP_configuration:
        print("Saving COMPUTER_PARAMETERS.h5 file.")
        print()
        with h5py.File('COMPUTER_PARAMETERS.h5','w') as hf:
            print(computerID)
            print(computerIP)
            print(computerCON)
            print()
            computerIP = [n.encode("ascii", "ignore") for n in computerIP]
            computerCON= [n.encode("ascii", "ignore") for n in computerCON]
            hf.create_dataset('ids', data=computerID)
            hf.create_dataset('ips', data=computerIP)
            hf.create_dataset('con', data=computerCON)    
    else:
        print("User did not wanted to save COMPUTER_PARAMETERS.h5 file.")
        print()
    
    cam_connected = True
    for cam_id in camera_id:
        ping_status = pingIP(cameraIP[int(cam_id)])
        if ping_status == False:
            cam_connected = False
        print("Camera "+str(cam_id)+" is working?: "+str(ping_status))
    print()
    cam_alive = True
    if cam_connected == True and isServerUp == True:
        for cam_id in camera_id:
            cam_id = int(cam_id)
            cam_status = testCamera(hostIP,cam_id,cameraIP)
            if cam_status == False:
                cam_alive = False
            print("Camera "+str(cam_id)+" is getting image and working with server?: "+str(cam_status))
    print()
    time.sleep(2)
    
    if isServerUp:
        killRedisServerOverClient(hostIP)
        time.sleep(1)
        isServerUp = pingRedisServer(hostIP)
        
    shell = runshellRedisServer(hostIP)
    time.sleep(1)
    isServerUp = pingRedisServer(hostIP)
    
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
            