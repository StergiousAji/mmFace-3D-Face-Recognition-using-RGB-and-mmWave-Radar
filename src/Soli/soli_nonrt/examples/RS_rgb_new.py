
import matplotlib.pyplot as plt
plt.ion()
import numpy as np
import time
from datetime import datetime
from skimage.util import view_as_blocks
import cv2

import pyrealsense2 as rs
#import argparse
import time

import os
from concurrent.futures import ProcessPoolExecutor
from invoke import run

import socket 

#%% References on Intel Realsense python wrapper

# https://intelrealsense.github.io/librealsense/python_docs/_generated/pyrealsense2.html
# 
# https://intelrealsense.github.io/librealsense/python_docs/_generated/pyrealsense2.frame.html#pyrealsense2.frame.data
# 
# https://intelrealsense.github.io/librealsense/python_docs/_generated/pyrealsense2.composite_frame.html

# https://github.com/IntelRealSense/librealsense/issues/1887#issuecomment-397529590

#%% Define functions to configure the IRS D435 and to grab frames

def config():
    print("Executing image Task on Process {}".format(os.getpid()))
    config = rs.config()
    #config.enable_stream(rs.stream.depth, 480, 270, rs.format.any, 60) #Params: stream, resolution_x, resolution_y, module, frame rate
    config.enable_stream(rs.stream.color, 640, 480, rs.format.any, 30) #Params: stream, resolution_x, resolution_y, module, frame rate
    
    #Create a pipeline, which is an object with the try_wait_for_frames() method
    global pipeline 
    pipeline = rs.pipeline()
    profile = pipeline.start(config)
    
    
    print("Configured")
    
def grab_frames():    
    #Declare variables to hold frame and system timestamps
    time_1 = 0
    time_2 = 0
    
    #Set the number of frames to capture. Ideally, this should match the length of time for which the 
    #SOLI will run. E.g. if the SOLI acquires frames over 20s, and the IRS is set to 15 FPS, we want 300 frames
    frame_number = 60*60
    
    #Instantiate empty array for depth frames
    depth_frames_list = np.zeros((frame_number,54,96))
    timestamps = np.zeros((frame_number,2))
    print("Grabbing frames...")
    try:   
       for i in range(frame_number):        
            #_,frames = pipeline.try_wait_for_frames()
            #if _ == True:

                #depth_frames_list[i] = depth_scale*np.asanyarray(frames[0].data).astype('float16') #the depth frames are in units of m       
                #timestamps[i,0] = frames.timestamp
                #timestamps[i,1] = time.time()
                #print('Frame',i,'device time difference',frames.timestamp-time_1,'system time difference',(time.time()-time_2)*1e3)
                #time_1 = frames.timestamp  
                #time_2 = time.time()
                
            frames = pipeline.wait_for_frames()
            image = np.asanyarray(frames[0].data)#the depth frames are in units of m      
            print(image.shape,image.dtype) 
            cv2.imshow('image',image)
            cv2.waitKey(1)
	    
    finally:
        pipeline.stop()
        np.save('../data/images.npy',depth_frames_list)
        np.save('../data/image_timestamps.npy',timestamps)
#config()  
#grab_frames()
#%%
def server():
    localIP     = "127.0.0.1"
    localPort   = 8080
    bufferSize  = 1024
    
     
    msgFromServer       = "Hello UDP Client"
    bytesToSend         = str.encode(msgFromServer)
    
    msgFromServer_2       = "SOLI ready"
    bytesToSend_2         = str.encode(msgFromServer_2)
    
    
    msgFromServer_3       = "Stop"
    bytesToSend_3         = str.encode(msgFromServer_3)
    

     # Create a datagram socket
    UDPServerSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)     
    # Bind to address and ip
    UDPServerSocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    UDPServerSocket.bind((localIP, localPort))
    print("UDP server up and listening")
     
    
    # Listen for incoming datagrams
    wait = True
    hello_message = 0
    ready_message = 0
    while(wait==True):
        bytesAddressPair = UDPServerSocket.recvfrom(bufferSize)
        message = bytesAddressPair[0]
        address = bytesAddressPair[1]
        clientMsg = "Message from Client:{}".format(message)
        clientIP  = "Client IP Address:{}".format(address)    
        print(clientMsg)
        print(clientIP)   
        
        # Sending a reply to client
        #UDPServerSocket.sendto(bytesToSend, address)
        
# =============================================================================
#         #Listen for a hello message. Once received from both clients, the device is to be configured.
#         if((message==b'Hello from soli client' or message==b'Hello from p2go client') and hello_message==0):
#             print('case 1')
#             hello_message = message
#         elif((message==b'Hello from soli client' and hello_message==b'Hello from p2go client') or (hello_message==b'Hello from soli client' and message==b'Hello from p2go client')):
#             print('case 2')
#             config()       
#         #Listen for a ready message. Once received from both clients, the SOLI device begins grabbing frames, 
#         #and so must the IRS
#         elif(message==b'p2go ready'):
#             print('case 3')
#             ready_message = message
#             address_1 = address            
#         elif((message==b'soli ready' and ready_message==b'p2go ready')):
#             print('case 4')
#             # Sending a reply to client 
#             UDPServerSocket.sendto(bytesToSend_2, address_1)           
#             grab_frames()          
#             #UDPServerSocket.sendto(bytesToSend_3, address)
#         #Listen for a complete message. Once received, the radar has finished grabbing frames,
#         #so the UDP socket can be closed.
#         elif(message==b'Complete'):
#             UDPServerSocket.close()
#             wait=False
#         time.sleep(0.001)
# =============================================================================
            #Listen for a hello message. Once received from both clients, the device is to be configured.
        if(message==b'Hello from soli client' and hello_message==0):
            print('case 1')
            hello_message = message
            print('case 2')
            config()       
        #Listen for a ready message. Once received from both clients, the SOLI device begins grabbing frames, 
        #and so must the IRS    
        elif(message==b'soli ready' ):
            print('case 4')
            # Sending a reply to client 
            #UDPServerSocket.sendto(bytesToSend_2, address_1)           
            grab_frames()          
            #UDPServerSocket.sendto(bytesToSend_3, address)
        #Listen for a complete message. Once received, the radar has finished grabbing frames,
        #so the UDP socket can be closed.
        elif(message==b'Complete'):
            UDPServerSocket.close()
            wait=False
        time.sleep(0.001)
 

#%% Execute
if __name__ == '__main__':
    server()