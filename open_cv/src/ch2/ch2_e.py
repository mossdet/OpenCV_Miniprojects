import numpy as np
import cv2 as cv2
import os
import matplotlib.pyplot as plt
 
"""
Scaling and rotation
"""

sep_ids_ls = [sep_idx for sep_idx, str_char in enumerate(__file__) if str_char == os.sep]
curr_path = __file__[0:sep_ids_ls[-1]] + os.sep
in_path = __file__[0:sep_ids_ls[-3]]+ os.sep + "Input_Images"+ os.sep
out_path = __file__[0:sep_ids_ls[-3]]+ os.sep + "Output_Images"+ os.sep
os.makedirs(out_path, exist_ok=True)

# Test image
test_img_path  = os.path.join(in_path, "players.jpg")
orig_img = cv2.imread(test_img_path, 1)
cv2.imshow("Original Image", orig_img)


"""
Scaling
"""
# generate an image that is half the size of the original in both dimensions
img_half = cv2.resize(orig_img, (0,0), fx=0.5, fy=0.5)

# Stretch original image to the dimensions 600 x 600 
img_stretch = cv2.resize(orig_img, (600,600))

# Stretch original image to the dimensions 600 x 600 bu this time using nearest interpolation mode
img_stretch_near = cv2.resize(orig_img, (600, 600), interpolation=cv2.INTER_LINEAR)

cv2.imshow("Half_Image", img_half)
cv2.imshow("Stretch_Image", img_stretch)
cv2.imshow("Stretch_NearInterpolated_Image", img_stretch_near)


"""
Rotation
"""
M = cv2.getRotationMatrix2D((orig_img.shape[1]/2, orig_img.shape[0]/2), -90, 1)
rotated_img = cv2.warpAffine(orig_img, M, (orig_img.shape[1], orig_img.shape[0]))
cv2.imshow("Rotated", rotated_img)

cv2.waitKey(0)
cv2.destroyAllWindows()

pass
