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
test_img_path  = os.path.join(in_path, "faces.jpeg")

orig_img = cv2.imread(test_img_path, 1)
height, width = orig_img.shape[0:2]
cv2.imshow("Original_Image", orig_img)

hsv_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2HSV)
h = hsv_img[:,:,0]
s = hsv_img[:,:,1]
v = hsv_img[:,:,2]

hsv_split_img = np.concatenate((h,s,v), axis=1)
cv2.imshow("Split_HSV_Image", hsv_split_img)


th = 40
ret, min_sat_img = cv2.threshold(s, th, 255, cv2.THRESH_BINARY)
cv2.imshow("Saturation_ThFilter_Image", min_sat_img)

th = 15
ret, max_hue_img = cv2.threshold(h, th, 255, cv2.THRESH_BINARY_INV) #THRESH_BINARY_INV
cv2.imshow("MaxHue_ThFilter_Image", max_hue_img)

final_img = cv2.bitwise_and(min_sat_img, max_hue_img)
cv2.imshow("Final_Image", final_img)


cv2.waitKey(0)
cv2.destroyAllWindows()
pass