from train_resnet50_cosine import EmbeddingNet
import torch
import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np
from PIL import Image
import math
import statistics
import time
import os

working_directory = "F:\CrystalDetector" # CHANGE THIS TO YOUR NEW FOLDER


# Choose whether to draw bounding boxes, velocity vector, or save images. 
draw_bounding_boxes = True
draw_velocity_vector =True
save_images = True

# Setup paths to video and working directory and detection model
video_source = r"video 4.mp4"
yolo_path = r'YOLO_best.pt'
embedder_path = r"cosine_epoch30.pth"
os.chdir(working_directory)

# Create a directory to store the modified images (With bounding boxes and velocity vector drawn on)
modified_images_dir = "modified_images"
os.makedirs(modified_images_dir, exist_ok=True)


curr_frame_idx = 0
crys_ids = {}
cap = cv2.VideoCapture(video_source)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


extractor_embed = EmbeddingNet()
checkpoint = torch.load(embedder_path, map_location=device)
extractor_embed.load_state_dict(checkpoint['state_dict'])

extractor_embed.to(device)
extractor_embed.eval()

print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")


model = YOLO(yolo_path)
model.to(device)
tracker = DeepSort(max_age=5,
                   n_init=3,
                   max_cosine_distance=0.0005,
                   max_iou_distance = 0.9,
                   embedder=None)


while True:
    ret, frame = cap.read()
    if not ret:
        break

    start = time.time()

    img = Image.fromarray(frame.astype(np.uint8))

    results = model(frame, conf=0.05, iou=0.2, verbose = False)[0]   # get first (and only) batch element

    # Build DeepSort detections list
    detections = []
    list_of_individual_crystal_image_as_PIL = []

    # each `box` is [x1, y1, x2, y2, conf, cls_id]
    for index in range(len(results.obb.xywhr)):

        # Get x,y(center of BB)
        x_min, y_min, x_max, y_max = results.obb.xyxy[index].cpu()
        w = float(x_max - x_min)
        h = float(y_max - y_min)

        detections.append(
            ([x_min, y_min, w, h],        # DeepSort wants [l,t,w,h]
             float(results.obb.conf[index].cpu()),           # confidence score
             int(results.obb.cls[index].cpu()))           # class id
        )

        # These are the corner coordinates of unrotated boxes surrounding each crystal
        x_min, y_min, x_max, y_max = map(int,results.obb.xyxy[index])

        # Crop an image of each crystal to be fed into DeepSORT.
        cropped_image_of_crystal = img.crop((x_min, y_min, x_max, y_max))
        list_of_individual_crystal_image_as_PIL.append(cropped_image_of_crystal)
    if len(list_of_individual_crystal_image_as_PIL) == 0:
        continue
    
    # Extracted features of each crystal as an array of vectors
    extracted_features_of_each_crystal = (extractor_embed(list_of_individual_crystal_image_as_PIL))


    tracks = tracker.update_tracks(
        detections,
        embeds=extracted_features_of_each_crystal,
        frame=frame             # This saves the frame as part of each crystal track, 
                                # but is not strictly necessary because the frames are drawn on while
                                # the program does detections.
    )


    # This counts the number of crystals currently shown in the frame as "confirmed"
    confirmed = [t for t in tracks if t.is_confirmed()]
    print(f"Frame {curr_frame_idx}: {len(confirmed)} detected crystals")
        
    # For each crystal track (a.k.a crystal ID) in the frame, calculate velocity and shape and size
    for track in confirmed:
        
        left_edge, top_edge, right_edge, bottom_edge = track.to_ltrb(orig=True)    # left, top, right, bottom

        w = right_edge - left_edge
        h = bottom_edge - top_edge
    
        crystal_id = track.track_id

        # Shape classification
        ratio = max(w,h)/min(w,h)
        if 1.0 < ratio < 1.25:
            shape = 0 # cube = 0
        elif 1.25 <= ratio < 5.0:
            shape = 1 # prism = 1
        else:
            shape = 2 # needle = 2

        if crystal_id in crys_ids.keys():
            
            crys_ids[crystal_id]["positions"].append((left_edge,top_edge))
            crys_ids[crystal_id]["size"].append(w*h)
            crys_ids[crystal_id]["shape"].append(shape)

            x1 = crys_ids[crystal_id]["positions"][-2][0]
            y1 = crys_ids[crystal_id]["positions"][-2][1]

            velocity = math.sqrt((left_edge-x1)**2 + (top_edge-y1)**2)
                
            angle = math.atan2(left_edge-x1 , top_edge-y1)

            crys_ids[crystal_id]["velocity"].append(velocity)
            crys_ids[crystal_id]["angle"].append(angle)


            if draw_bounding_boxes:
                
                box_color  = (255, 0, 0) #BGR
                text_color = (255, 0, 0) #BGR
                line_color = (255, 0, 0) #BGR

                line_thickness = 3
                length_scale_factor = 10
                velocity_vector_start_x = int(left_edge + w/2)
                velocity_vector_start_y = int(top_edge + h/2)
                
                velocity_vector_end_x = int(velocity_vector_start_x+velocity*length_scale_factor*math.sin(angle))
                velocity_vector_end_y = int(velocity_vector_start_y+velocity*length_scale_factor*math.cos(angle))

                cv2.rectangle(frame, (int(left_edge), int(top_edge)), (int(left_edge+w), int(top_edge+h)), box_color, 2)
                cv2.putText(frame, f"ID {crystal_id}", (int(left_edge), int(top_edge)-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
                
                if draw_velocity_vector:
                    cv2.line(frame, (velocity_vector_start_x, velocity_vector_start_y),
                                    (velocity_vector_end_x  , velocity_vector_end_y), line_color, line_thickness) 

        else:
            crys_ids[crystal_id] = {"positions":[(left_edge,top_edge)],
                                    "velocity": [],
                                    "angle":[],
                                    "size":[w*h],
                                    "shape":[shape]}



    if save_images: 
        cv2.imwrite(modified_images_dir + rf"\frame_{curr_frame_idx:04d}.jpg", frame)

    if cv2.waitKey(1) == 27:  # ESC to quit
        break
    
    curr_frame_idx += 1

print("\nGenerating statistics about the detected crystals. Please Wait...\n")

side_length = []
sizes_list_hist = []
size = []
shape = []
velocity = []
unique_crystals = 0
for crystal_id in crys_ids:
    if (len(crys_ids[crystal_id]["positions"]) > 3):

        sizes_list_hist.extend(crys_ids[crystal_id]["size"])

        for area in sizes_list_hist:
            area = area/1.37/1.37
            side_length.append(math.sqrt(area))


        for vel in crys_ids[crystal_id]["velocity"]:
            velocity.append(abs(vel))

        size.append(crys_ids[crystal_id]["size"])
        shape.append(crys_ids[crystal_id]["shape"])
        
        unique_crystals += 1
print("Unique Crystals "+str(unique_crystals))




avg_vel = sum(velocity) / len(velocity)
std_dev = statistics.stdev(velocity)

print("Average Velocity: " + str(avg_vel))
print("Std Velocity: " + str(std_dev))
