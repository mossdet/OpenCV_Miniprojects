import numpy as np
import cv2 as cv2
import os
import matplotlib.pyplot as plt
 
"""
Object Detection, Eye Detection
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

xml_filepath = in_path+"haarcascade_eye.xml"
eye_cascade = cv2.CascadeClassifier(xml_filepath)

eyes = eye_cascade.detectMultiScale(gray_img, scaleFactor=1.05, minNeighbors=3, minSize=(1,1), maxSize=(30,30))

print(len(eyes))

for (x,y,w,h) in eyes:
    point = (int(x+w/2),int(y+h/2))
    cv2.circle(orig_img, point, int(w/3), (0,255, 0), thickness=2)
    #cv2.circle(frame, point, radius, color, line_width)

cv2.imshow("Image", orig_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
pass