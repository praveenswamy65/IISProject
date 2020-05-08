import cv2
import numpy as np
import ipywidgets as widgets
from matplotlib import pyplot as plt
from IPython.display import display

#Output file to be written
output_file_name = "Output.mp4"
backend = cv2.CAP_ANY
fourcc_code = cv2.VideoWriter_fourcc(*"X264")
fps = 12
frame_size = (2*640, 480)
output_video = cv2.VideoWriter(output_file_name, backend, fourcc_code, fps, frame_size)

#Input file to be read
video_reader = cv2.VideoCapture("open_palm.webm")
ret, frame = video_reader.read()

#Threshold range 
lower = [3, 88, 138]
upper = [255, 255, 255]
lower_hsv = lower
upper_hsv = upper

disp = widgets.Image()
display(disp)

i=1
num_frames=0
while ret:
        ret, frame = video_reader.read() 
        if not ret:
            continue
        num_frames+=1
        i=i+1
   
   # color threshold using HSV
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)   
        mask = cv2.inRange(frame_hsv, np.array(lower_hsv), np.array(upper_hsv))
        masked_hsv = cv2.bitwise_and(frame, frame, mask = mask)
    
    # draw contours from HSV thresholding
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
        max_idx = 0
        max_val = 0
        for idx, c in enumerate(contours):
            if cv2.contourArea(c) > max_val:
                max_idx = idx
                max_val = cv2.contourArea(c)
        
    # compute the center of the biggest contour
        cv2.drawContours(masked_hsv, contours[max_idx], -1, (255, 255, 255), 2)
        M = cv2.moments(contours[max_idx])
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        cv2.circle(masked_hsv, (cX, cY), 7, (255, 255, 255), -1)
        cv2.putText(masked_hsv, "center", (cX - 20, cY - 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(masked_hsv, "HSV", (5, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4) 
        cv2.putText(frame, "Original", (5, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
        cv2.drawContours(frame, contours[max_idx], -1, (0, 0, 255), 2)
        cv2.circle(frame, (cX, cY), 7, (0, 0, 255), -1)
        cv2.putText(frame, "center", (cX - 20, cY - 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        display_image = cv2.imencode('.png', frame)[1].tostring()
        disp.value = display_image
        continue
        frame_stacked = np.hstack([frame, masked_hsv])
        output_video.write(frame_stacked)  
video_reader.release()
output_video.release()
