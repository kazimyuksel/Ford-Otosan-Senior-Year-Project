# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 14:15:24 2019

@author: Kazım
"""

import socket    
hostname = socket.gethostname()    
IPAddr = socket.gethostbyname(hostname)    
print("Your Computer Name is:" + hostname)    
print("Your Computer IP Address is:" + IPAddr)

import os

hostname = "192.168.1.2" #example
response = os.system("ping -n 1 " + hostname)

hostname = "192.168.1.112" #example
a = os.popen("ping -n 1 " + hostname).read()
a = list(a)
a[72]
for i in range(1,256):
    hostname = "192.168.1."+str(i) #example
    a = os.popen("ping -n 1 " + hostname).read()
    a = list(a)
    if a[72] == 'D':
        print(0)
    else:
        print(1)
        
import subprocess
import ipaddress
from subprocess import Popen, PIPE

for ip in ipaddress.IPv4Network('192.168.1.0/24'):
    ip=str(ip)
    toping = Popen(["ping","-n","1", ip], stdout=PIPE)
    output=toping.communicate()[0].decode("utf-8")
    word = "Destination"
    hostalive = word in output
    if hostalive ==0:
        print (ip,"is reachable")
    else:
        print(ip,"is unreachable")
     
word = "Destination"
mystring = a
if word in mystring: 
   print("success")
print (socket.getfqdn("192.168.1.0"))
