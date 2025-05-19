import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# imports
import cv2 as cv 
import time  
import torch
import keyboard
import numpy as np
import matplotlib.pyplot as plt
plt.ion()
from ultralytics import YOLO
from models.models import FastDepthV2
from torchvision import transforms
from PIL import Image

# YOLO model
yolo = YOLO("yolov8n.pt")

# FastDepth model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FastDepthV2()
checkpoint = torch.load(r"E:\INTERNSHIP\AI_Projects\envs\obj_detect_env\FastDepth\Weights\FastDepthV2_L1GN_Best.pth", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device).eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def show_closest_object_depth(depth_maps):
    closest_label = None
    closest_map = None
    min_mean_depth = float('inf')

    for label, dmap in depth_maps.items():
        if dmap.size == 0:
            continue
        center = dmap[dmap.shape[0]//4:3*dmap.shape[0]//4, dmap.shape[1]//4:3*dmap.shape[1]//4]
        mean_depth = center.mean()
        if mean_depth < min_mean_depth:
            min_mean_depth = mean_depth
            closest_label = label
            closest_map = dmap

    plt.figure(f"Closest: {closest_label}")
    plt.imshow(closest_map, cmap='inferno')
    plt.title(f"{closest_label} (Mean Depth: {min_mean_depth:.2f})")
    plt.colorbar()
    plt.axis('off')
    plt.show(block=False)
    plt.pause(0.001)

# Variables
previous_time = 0  
object_detected_time = None  
object_locked_time = None  
prompted = False  
object_locked = False  
frame_sent = 0 
detection_enabled = True

# Capture
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)  
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)  
cap.set(cv.CAP_PROP_FPS, 30)

# Loop
while True:
    i=0
    for i in range(0, 1):
        ret, frame = cap.read()  
        if not ret:
            break

        flip_frame = cv.flip(frame, 1)  
        resized_Frame = cv.resize(flip_frame, (640, 480))  

        # ROI
        x, y = 170, 90
        width, height = 300, 300
        ROI = resized_Frame[y:y+height, x:x+width]  
        cv.rectangle(resized_Frame, (x, y), (x+width, y+height), (0, 255, 0), 2)

        if frame_sent != 1:  
            results = yolo.track(ROI, stream=True, verbose=False)  
            object_in_rectangle = False  
            detected_objects = set()
            depth_maps = {}

            for result in results:
                for box in result.boxes: 
                    if box.conf[0] > 0.6:  
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        margin = 10
                        if (
                            x1 >= -margin and y1 >= -margin and
                            x2 <= width + margin and y2 <= height + margin
                        ):
                            object_in_rectangle = True  
                            if object_detected_time is None:  
                                object_detected_time = time.time()  

                        cls = int(box.cls[0]) 
                        class_name = result.names[cls]
                        detected_objects.add(class_name) 
                        object = cv.rectangle(ROI, (x1, y1), (x2, y2), (0, 255, 0), 2)  
                        cv.putText(ROI, f'{class_name} {box.conf[0]:.2f}', (x1, y1), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                        # Object depth map
                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(width, x2)
                        y2 = min(height, y2)
                        object_crop = ROI[y1:y2, x1:x2]
                        if object_crop.size != 0:
                            input_tensor_obj = transform(object_crop).unsqueeze(0).to(device)
                            with torch.no_grad():
                                depth_output_obj = model(input_tensor_obj)
                                depth_map_obj = depth_output_obj.squeeze().cpu().numpy()
                            depth_maps[class_name] = depth_map_obj

            if object_in_rectangle:  
                duration = time.time() - object_detected_time if object_detected_time else 0 
                if duration > 1: 
                    object_locked = True
                    cv.rectangle(resized_Frame, (x, y), (x+width, y+height), (0, 0, 255), 2)  
                    if object_locked_time is None:
                        object_locked_time = time.time()    
            else:
                object_detected_time = None  
                object_locked_time = None  
                object_locked = False  
                prompted = False  

            if object_locked and object_locked_time and not prompted and (time.time() - object_locked_time > 1):
                prompted = True
                print("Detected objects:", ", ".join(detected_objects))
                cv.imshow("Depth analysis of ROI", ROI)
                frame_sent = 1

                # Depth map of the full ROI
                input_tensor_roi = transform(ROI).unsqueeze(0).to(device)
                with torch.no_grad():
                    depth_output_roi = model(input_tensor_roi)
                    depth_map_roi = depth_output_roi.squeeze().cpu().numpy()

                plt.figure("Depth Map of ROI")
                plt.imshow(depth_map_roi, cmap='inferno')
                plt.title("Depth Map: Full ROI")
                plt.colorbar()
                plt.axis('off')
                plt.show(block=False)
                plt.pause(0.001)


                # Show closest object's depth map only
                show_closest_object_depth(depth_maps)

        # Crosshair
        cv.line(resized_Frame, (320, 245), (320, 235), (0, 255, 0), 1)
        cv.line(resized_Frame, (315, 240), (325, 240), (0, 255, 0), 1) 

        # FPS
        current_time = time.time()  
        fps = 1 / (current_time - previous_time)  
        previous_time = current_time
        cv.putText(resized_Frame, f'FPS: {int(fps)}', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  

        # Display
        cv.imshow("Object Detection", resized_Frame)  

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()  
cv.destroyAllWindows()

