import sys

import argparse

sys.path.insert(0,'home/agrotech/deep-gauge-master/Reading_Circular_Analogue_Gauges_using_Digital_Image_Processing')

import torch

import cv2

import time

import re

import numpy as np

from ultralytics import YOLO

import math

from typing import List

model_path = '/home/agrotech/deep-gauge-master/Reading_Circular_Analogue_Gauges_using_Digital_Image_Processing/model_weights/pose_gauge.pt'

theta_A=0

theta_B=0



### -------------------------------------- function to run detection ---------------------------------------------------------

def detectx (frame, model):

 frame = [frame]

 print(f"[INFO] Detecting. . . ")

 results = model(frame)

 #results.show()

 print(results.xyxyn[0])

 print(results.xyxyn[0][:, -1])

 print(results.xyxyn[0][:, :-1])

 labels, cordinates = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

 return labels, cordinates



### ------------------------------------ to plot the BBox and results --------------------------------------------------------

def plot_boxes(results, frame,classes):



 """

 --> This function takes results, frame and classes

 --> results: contains labels and coordinates predicted by model on the given frame

 --> classes: contains the strting labels



 """

 labels, cord = results

 n = len(labels)

 x_shape, y_shape = frame.shape[1], frame.shape[0]



 print(f"[INFO] Total {n} detections. . . ")

 print(f"[INFO] Looping through all detections. . . ")



 ### looping through the detections

 for i in range(n):

 row = cord[i]

 if row[4] >= 0.37: ### threshold value for detection. We are discarding everything below this value

 print(f"[INFO] Extracting BBox coordinates. . . ")

 x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape) ## BBOx coordniates

 text_d = classes[int(labels[i])]

 print(text_d)

 coords = [x1,y1,x2,y2]

 plate_num, plantnum1 = calculate_pointer_angle(img = frame, coords= coords)

 minimum, maximum, argument=gauge_read(text_d) 

 

 if argument=='Gauge not in Dictionary':

 text_d='Gauge not in Dictionary'

 

 print(f"minium is: {minimum:.1f}")

 print(f"maximum is: {maximum:.1f}")



 value=minimum+(plate_num/plantnum1)*maximum



 print(f"Gauge reading is {value:.1f}" ) 



 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) ## BBox

 cv2.rectangle(frame, (x1, y1-20), (x2, y1), (0, 255,0), -1) ## for text label background

 cv2.putText(frame, f"{value:.1f}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255), 2)

 cv2.putText(frame, f"{text_d}", (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, .75,(255,255,0), 1) 

 return frame




def calculate_pointer_angle(img, coords):# Calculates the Pointer Angle 

 # separate coordinates from box

 xmin, ymin, xmax, ymax = coords

 nplate = img[int(ymin):int(ymax), int(xmin):int(xmax)] ### cropping the gauge from the whole image

 #Detect the gauge and get the pointer angle

 model1 = YOLO(model_path)

 results = model1(img)

 theta1 = None # Initialize with a default value

 theta2= None # Initialize with a default value



 for result in results:

 keypoints_set = result.keypoints.xy.cpu().numpy()

 print(keypoints_set)



 # Check if keypoints_set is empty

 if keypoints_set.size == 0:

 print('No Key Point Detected')

 theta1=0

 theta2=45

 break

 # Check if there are at least 4 keypoints in the set

 if keypoints_set.shape[1] < 4:

 print('Not enough keypoints detected')

 break



 print(keypoints_set[0])

 A, B, C , D = keypoints_set[0][0], keypoints_set[0][1], keypoints_set[0][3] , keypoints_set[0][2]

 print(keypoints_set[0])

 print(f"A is {A[1]}")

 print(f"B is {B[1]}")

 print(f"C is {C[1]}")

 

 theta1 = compute_angle(A, B, C)

 theta2 = 360-compute_angle(B, D, C)

 print(f"Angle between keypoints A, B, and C is: {theta1:.2f} degrees")

 print(f"Angle between keypoints B, C, and D is: {theta2:.2f} degrees")



 # for indx, keypoint in enumerate([A, B, C]):

 # cv2.putText(img, str(keypoint_indx + indx), (int(keypoint[0]), int(keypoint[1])),

 # cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)




 return theta1, theta2





def compute_angle(A, B, C):

 # Vectors for BA and BC

 a = [A[0] - B[0], A[1] - B[1]]

 b = [C[0] - B[0], C[1] - B[1]]

 c = [A[0] - C[0], A[1] - C[1]]



 # Magnitudes (lengths) of BA, BC, and AC

 ma = math.sqrt(a[0]**2 + a[1]**2)

 mb = math.sqrt(b[0]**2 + b[1]**2)

 mc = math.sqrt(c[0]**2 + c[1]**2)



 print(ma)

 print(mb)

 print(mc)

 # Using the law of cosines to find the angle at B

 cosB = (ma**2 + mc**2 - mb**2) / (2 * ma * mc)

 

 # Angle in radians

 angle = math.acos(cosB)



 # Convert angle to degrees

 angle_deg = math.degrees(angle)



 return angle_deg




### ---------------------------------------------- Main function -----------------------------------------------------



def main(img_path=None):

 print(f"[INFO] Loading V5 model... ")

 ## loading the custom trained model

 #model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5x.pt',force_reload=True) ## if you want to download the git repo and then run the detection

 model = torch.hub.load('/home/agrotech/deep-gauge-master/Reading_Circular_Analogue_Gauges_using_Digital_Image_Processing/yolov5-master/yolov5/yolov5', 'custom',source ='local', path='demo_20feb_v1.pt',force_reload=True)

 #model = torch.hub.load('ultralytics/yolov5', 'custom', path='demo_20feb_v1.pt',force_reload=True)

 #model = torch.hub.load('./yolov5-master', 'custom', source ='local', path='demo_20feb_v1.pt',force_reload=True) ### The repo is stored locally

 classes = model.names ### class names in string format



 ### --------------- for detection on image --------------------

 if img_path != None:

 print(f"[INFO] Working with image: {img_path}")

 img_out_name = f"./output/result_{img_path.split('/')[-1]}"

 frame = cv2.imread(img_path) ### Reading the image

 frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

 results = detectx(frame, model = model) ### DETECTION HAPPENING HERE 

 frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)

 frame = plot_boxes(results, frame,classes = classes)

 #cv2.namedWindow("img_only", cv2.WINDOW_NORMAL) ## Creating a free windown to show the result



 while True:

 frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)

 cv2.imwrite(f"output.jpg",frame) ## if you want to save he output result.

 break

 



def gauge_read(text_d):

 # Define a dictionary to store gauge names and their respective min, max, and description values

 gauge_limits = {

 'Pressure Gauge2': (0, 16, None),

 'Pressure Gauge3': (0, 15, None),

 'Pressure Gauge4': (0, 10, None),

 'Pressure Gauge1': (0, 28, None),

 'Fire Pressure Gauge1': (0, 15, None),

 'SG6 Pressure Gauge1': (0, 950, None)

 }



 # Fetch values from the dictionary using get() method; 

 # defaults to (0, 0, "Gauge not found") if key not found

 result = gauge_limits.get(text_d, (0, 0, "Gauge not in Dictionary"))



 print(text_d)



 return result








if __name__ == "__main__":

 parser = argparse.ArgumentParser(description='Process some images')

 parser.add_argument('--img', type=str, help='The path of the image file to process')

 #parser.add_argument('--vid', type=str, help='The path of the video file to process')

 args = parser.parse_args()

 main(img_path=args.img)



### ------------------- calling the main function-------------------------------

#main(vid_path="./test_images/VID20230220184956.mp4",vid_out="vid_4.mp4") ### for custom video

#main(vid_path=0,vid_out="./test_images/man1.mov") #### for webcam

#main(vid_path=0)

#main(img_path="./station2.png") ## for image

#'rtsp://admin:SMSS@2022@172.29.98.172:554'

#url= "rtsp://admin:SMSS@2022@172.29.98.172:554" 

#main(vid_path= url)