# REFERENCES
# https://intelrealsense.github.io/librealsense/python_docs/_generated/pyrealsense2.html
# https://intelrealsense.github.io/librealsense/python_docs/_generated/pyrealsense2.frame.html#pyrealsense2.frame.data
# https://intelrealsense.github.io/librealsense/python_docs/_generated/pyrealsense2.composite_frame.html
# https://github.com/IntelRealSense/librealsense/issues/1887#issuecomment-397529590

# https://dev.intelrealsense.com/docs/python2

import sys
import os
import time
import pyrealsense2 as rs
import numpy as np
import cv2
import socket
from experiment import experiment

SUBJECT_PATH = ""

def config():
    print("Executing image Task on Process {}".format(os.getpid()))
    config = rs.config()

    # Params: stream, resolution_x, resolution_y, module, frame rate
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    # Create a pipeline, which is an object with the try_wait_for_frames() method
    global pipeline 
    pipeline = rs.pipeline()
    profile = pipeline.start(config)

    # Find the scaling from depth values of pixels to metres
    depth_sensor = profile.get_device().first_depth_sensor()
    global depth_scale
    depth_scale = depth_sensor.get_depth_scale()

    depth_sensor.set_option(rs.option.min_distance, 0)
    depth_sensor.set_option(rs.option.enable_max_usable_range, 0)
    depth_sensor.set_option(rs.option.confidence_threshold, 1.0)
    depth_sensor.set_option(rs.option.laser_power, 100)
    depth_sensor.set_option(rs.option.receiver_gain, 9)
    
    print("Configured")

def grab_frames(save=True):
    # Set the number of frames to capture proportional to FPS. 
    frames = 10
    if not save:
        frames = 120
    # Height = Rows, Width = Columns
    colour_frames_list = np.zeros((frames, 480, 640, 3), np.float16)

    # rs.align allows us to perform alignment of depth frames to others frames
    align = rs.align(rs.stream.color)
    
    timestamps = np.zeros((frames, 2))

    print("Grabbing frames...")
    try:   
        for i in range(frames):        
            frames = pipeline.wait_for_frames()

            aligned_frames = align.process(frames)
            colour_frame = aligned_frames.get_color_frame()
            
            colour_image = np.asanyarray(colour_frame.data)
            # Reverse BGR to RGB
            colour_frames_list[i] = colour_image.astype(np.float16)[:,:,::-1]

            timestamps[i, 0] = frames.timestamp
            timestamps[i, 1] = time.time()
            print(f"\nFrame {i}\nDevice Time Difference: {frames.timestamp - timestamps[i-1, 0]}\nSystem Time Difference: {(time.time() - timestamps[i-1, 1])*1e3}")

            cv2.imshow("RealSense", colour_image)
            cv2.waitKey(1)
    finally:
        if save:
            pipeline.stop()
            print("\nSaving frames...")
            np.save(f"{SUBJECT_PATH}_colour.npy", colour_frames_list)
            np.save(f"{SUBJECT_PATH}_timestamps.npy", timestamps)

def server():
    localIP     = "127.0.0.1"
    localPort   = 8080
    bufferSize  = 1024
    
    # Create a datagram socket
    UDPServerSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)     
    # Bind to address and ip
    UDPServerSocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    UDPServerSocket.bind((localIP, localPort))
    print("UDP server up and listening")
    
    # Listen for incoming datagrams
    wait = True
    hello_message = 0
    while wait:
        bytesAddressPair = UDPServerSocket.recvfrom(bufferSize)
        message = bytesAddressPair[0]
        address = bytesAddressPair[1]
        clientMsg = "Message from Client:{}".format(message)
        clientIP = "Client IP Address:{}".format(address)
        print(clientMsg)
        print(clientIP)   
        # Listen for a hello message. Once received from both clients, the device is to be configured.
        if (message == b'Hello from soli client' and hello_message == 0):
            print('case 1')
            hello_message = message
            print('case 2')
            config()
            grab_frames(save=False)
            time.sleep(1)
        # Listen for a ready message. Once received from both clients, the SOLI device begins grabbing frames, and so must the IRS    
        elif (message == b'soli ready' ):
            print('case 4')        
            grab_frames()          
        # Listen for a complete message. Once received, the radar has finished grabbing frames, so the UDP socket can be closed.
        elif (message == b'Complete'):
            UDPServerSocket.close()
            wait = False
        time.sleep(0.001)

def subject_exists():
    return os.path.exists(f"{SUBJECT_PATH}_depth.npy") or os.path.exists(f"{SUBJECT_PATH}_colour.npy") or os.path.exists(f"{SUBJECT_PATH}_timestamps.npy")

def make_dir(dirpath):
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)

if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1].isdigit() and sys.argv[2].isdigit():
        SUBJECT_PATH = f"./data/{sys.argv[1]}/{sys.argv[1]}-{sys.argv[2]}"
        if not subject_exists():
            make_dir(f"./data/{sys.argv[1]}/")
            print(f"SUBJECT: {sys.argv[1]} | EXPERIMENT: {sys.argv[2]} - {experiment[int(sys.argv[2]) // 5][int(sys.argv[2]) % 5]}")
            server()
        else:
            print(f"ERROR: SUBJECT {sys.argv[1]} DATA FILES OF EXPERIMENT {sys.argv[2]} ALREADY EXIST")
    else:
        print("ERROR: MUST INPUT VALID SUBJECT IDENTIFIER AND EXPERIMENT NUMBER")