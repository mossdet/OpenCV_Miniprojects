import numpy as np
import cv2 as cv2
import os
import matplotlib.pyplot as plt
 
"""
Object Detection, Adaptive Thresholding
"""

sep_ids_ls = [sep_idx for sep_idx, str_char in enumerate(__file__) if str_char == os.sep]
curr_path = __file__[0:sep_ids_ls[-1]] + os.sep
in_path = __file__[0:sep_ids_ls[-3]]+ os.sep + "Input_Images"+ os.sep
out_path = __file__[0:sep_ids_ls[-3]]+ os.sep + "Output_Images"+ os.sep
os.makedirs(out_path, exist_ok=True)

# Test image
test_img_path  = os.path.join(in_path, "sudoku.png")

orig_img = cv2.imread(test_img_path, 0)
height, width = orig_img.shape[0:2]
cv2.imshow("Original_BW_Image", orig_img)

# Faster thresholded binary image
th = 70
ret, th_basic_img = cv2.threshold(orig_img, th, 255, cv2.THRESH_BINARY)
cv2.imshow("Simple_Th_Image", th_basic_img)

neighborhood_param = 115
th_adapt_img = cv2.adaptiveThreshold(orig_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, neighborhood_param, 1)
cv2.imshow("Adaptive_Th_Image", th_adapt_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
pass