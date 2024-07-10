import numpy as np
import cv2 as cv2
import os
import matplotlib.pyplot as plt
 
"""
Custom Interfaces with Video Capture
"""

sep_ids_ls = [sep_idx for sep_idx, str_char in enumerate(__file__) if str_char == os.sep]
curr_path = __file__[0:sep_ids_ls[-1]] + os.sep
in_path = __file__[0:sep_ids_ls[-3]]+ os.sep + "Input_Images"+ os.sep

# CV2 handles the communication with the camera
capture = cv2.VideoCapture(0)

# assuming that the video camera has an 8 bit resolution 
color = (0, 255, 0)

# if -1, the circle would be filled, instead of having a line of thickness
line_width = 3
radius = 100
point = (0,0)


xml_filepath = in_path+"haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(xml_filepath)

xml_filepath = in_path+"haarcascade_eye.xml"
eye_cascade = cv2.CascadeClassifier(xml_filepath)

while(True):

    ret, frame = capture.read()
    frame = cv2.resize(frame, (0,0), fx=2.5, fy=2)
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.15, minNeighbors=10, minSize=(100, 100))
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255, 0), thickness=2)

    eyes = eye_cascade.detectMultiScale(gray_img, scaleFactor=1.05, minNeighbors=30, minSize=(50, 50), maxSize=(150, 150))
    for (x,y,w,h) in eyes:
        point = (int(x+w/2),int(y+h/2))
        cv2.circle(frame, point, int(w/4), (0,255, 0), thickness=2)


    cv2.imshow("Frame", frame)

    ch = cv2.waitKey(1)
    if ch & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()