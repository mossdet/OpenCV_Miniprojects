import numpy as np
import cv2 as cv2
import os
import matplotlib.pyplot as plt
 
"""
Custom Interfaces with Video Capture
"""

# CV2 handles the communication with the camera
capture = cv2.VideoCapture(0)

# assuming that the video camera has an 8 bit resolution 
color = (0, 255, 0)

# if -1, the circle would be filled, instead of having a line of thickness
line_width = 3
radius = 100
point = (0,0)

# Callback function
def click(event, x, y, flags, param):
    global point, pressed
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Pressed", x,y)
        point = (x,y)

# Be sure that the window name here is the same as in the while loop
cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", click)

while(True):

    ret, frame = capture.read()

    frame = cv2.resize(frame, (0,0), fx=2.5, fy=2)
    #point = (int(frame.shape[1]/2), int(frame.shape[0]/2))
    cv2.circle(frame, point, radius, color, line_width)

    cv2.imshow("Frame", frame)

    ch = cv2.waitKey(1)
    if ch & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()