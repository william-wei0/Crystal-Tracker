#Get training images

import cv2
import os 
import numpy as np 
from tqdm import tqdm 
from collections import deque

base_dir = os.path.expanduser("./")

video_filename = '../Videos/video4.mp4'
video_path = os.path.join(base_dir, video_filename)

output_dir = os.path.join(base_dir, 'VideoinFrames')
os.makedirs(output_dir, exist_ok=True)

output_key_frame_dir = os.path.join(base_dir, 'KeyFrames')  # directory to save frames with contours
os.makedirs(output_key_frame_dir, exist_ok=True)

def get_frames():
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): 
        print(f"[Error] Could not open video file at {video_path}")
        exit()

    frame_count = 0  
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


    for _ in tqdm(range(total_frames), desc="Extracting frames"):
        ret, frame = cap.read() 
        if not ret:
            break 

        frame_path = os.path.join(output_dir, f'frame_{frame_count:04d}.jpg')

        # save the frame as an image file
        cv2.imwrite(frame_path, frame)
        frame_count += 1  # increment the frame counter

    cap.release()  
    print(f"Finished saving {frame_count} frames to {output_dir}")  
    return frame_count


def move_key_frames(total_frames):
    for frame_count in range(int(total_frames/100)):
        frame_path = os.path.join(output_dir, f'frame_{frame_count*100:04d}.jpg')
        dst_path = os.path.join(output_key_frame_dir, f'frame_{frame_count*100:04d}.jpg')
        if (os.path.exists(frame_path)):
            os.rename(frame_path,dst_path)
        else:
            print(f"Frame at {frame_path} does not exist.")
