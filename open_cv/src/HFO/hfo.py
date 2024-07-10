import numpy as np
import cv2 as cv2
import os
import matplotlib.pyplot as plt
 
"""
Object Detection, Contour detection
"""

sep_ids_ls = [sep_idx for sep_idx, str_char in enumerate(__file__) if str_char == os.sep]
curr_path = __file__[0:sep_ids_ls[-1]] + os.sep
in_path = __file__[0:sep_ids_ls[-3]]+ os.sep + "Input_Images"+ os.sep
out_path = __file__[0:sep_ids_ls[-3]]+ os.sep + "Output_Images"+ os.sep
os.makedirs(out_path, exist_ok=True)


# HFO Spectrogram
test_img_path  = os.path.join(in_path, "HFO.png")
color_img = cv2.imread(test_img_path, 1)
color_img = cv2.resize(color_img, (1000, 600), interpolation=cv2.INTER_LINEAR)
height, width = color_img.shape[0:2]
cv2.imshow("Original_Image", color_img)

# Extract channels
b = color_img[:,:,0]
g = color_img[:,:,1]
r = color_img[:,:,2]

# Only Red channel image
r_hfo_img = cv2.merge((r,r,r))
cv2.imshow("RED_HFO", r_hfo_img)

# To Gray scale
gray_hfo_img = r#cv2.cvtColor(r_hfo_img, cv2.COLOR_RGB2GRAY)
cv2.imshow("Gray_HFO", r_hfo_img)

# Simple Thresholding
th = 99.99
ret, th_hfo_img = cv2.threshold(gray_hfo_img, th, 255, cv2.THRESH_BINARY)
cv2.imshow("Simple_Th_HFO", th_hfo_img)

# Adaptive Thresholding
neighborhood_param = 115
adapt_th_hfo_img = cv2.adaptiveThreshold(gray_hfo_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, neighborhood_param, 1)
#cv2.imshow("Adaptive_Th_HFO", adapt_th_hfo_img)

# Find Contours
# Stretch original image to the dimensions 600 x 600 bu this time using nearest interpolation mode
contours, hierarchy = cv2.findContours(th_hfo_img, cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)

color_img2 = color_img.copy()
index = -1
thickness = 4
color = (255, 0, 255)
maxLevel = 0
cv2.drawContours(image=color_img2, contours=contours, contourIdx=index, color=color, thickness=thickness, hierarchy=hierarchy, maxLevel=maxLevel)
cv2.imshow("Contours", color_img2)

cv2.waitKey(0)
cv2.destroyAllWindows()

pass