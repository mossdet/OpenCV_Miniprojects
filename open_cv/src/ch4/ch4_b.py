import numpy as np
import cv2 as cv2
import os
import matplotlib.pyplot as plt
 
"""
Object Detection, Face Detection
"""

sep_ids_ls = [sep_idx for sep_idx, str_char in enumerate(__file__) if str_char == os.sep]
curr_path = __file__[0:sep_ids_ls[-1]] + os.sep
in_path = __file__[0:sep_ids_ls[-3]]+ os.sep + "Input_Images"+ os.sep
out_path = __file__[0:sep_ids_ls[-3]]+ os.sep + "Output_Images"+ os.sep
os.makedirs(out_path, exist_ok=True)

# Test image
img_path  = os.path.join(in_path, "faces.jpeg")
orig_img = cv2.imread(img_path, 1)
gray_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)

#cv2.imshow("Frame", orig_img)

xml_filepath = in_path+"haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(xml_filepath)

faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.15, minNeighbors=3, minSize=(40,40))

print(len(faces))

for (x,y,w,h) in faces:
    cv2.rectangle(orig_img, (x,y), (x+w, y+h), (0,255, 0), thickness=2)

cv2.imshow("Image", orig_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
pass