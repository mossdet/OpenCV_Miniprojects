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

# Test image
test_img_path  = os.path.join(in_path, "detect_blob.png")
#test_img_path  = os.path.join(in_path, "HFO.png")

orig_img = cv2.imread(test_img_path, 1)
height, width = orig_img.shape[0:2]

gray_img = cv2.cvtColor(orig_img, cv2.COLOR_RGB2GRAY)
neighborhood_param = 115
th_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, neighborhood_param, 1)

cv2.imshow("Original_Image", orig_img)
cv2.imshow("Gray_Image", gray_img)
cv2.imshow("TH_Image", th_img)

contours, hierarchy = cv2.findContours(th_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

img2 = orig_img.copy()

index = -1
thickness = 4
color = (255, 0, 255)
cv2.drawContours(img2, contours, index, color=color, thickness=thickness)

cv2.imshow("Contours", img2)

cv2.waitKey(0)
cv2.destroyAllWindows()
pass