import numpy as np
import cv2 as cv2
import os
import matplotlib.pyplot as plt
 
"""
Object Detection, Simple Thresholding
"""

sep_ids_ls = [sep_idx for sep_idx, str_char in enumerate(__file__) if str_char == os.sep]
curr_path = __file__[0:sep_ids_ls[-1]] + os.sep
in_path = __file__[0:sep_ids_ls[-3]]+ os.sep + "Input_Images"+ os.sep
out_path = __file__[0:sep_ids_ls[-3]]+ os.sep + "Output_Images"+ os.sep
os.makedirs(out_path, exist_ok=True)

# Test image
test_img_path  = os.path.join(in_path, "detect_blob.png")

orig_img = cv2.imread(test_img_path, 0)
height, width = orig_img.shape[0:2]
cv2.imshow("Original_BW_Image", orig_img)


# Slow Thresholded binary image
binary_img = np.zeros([height, width, 1], 'uint8')

th = 85

for row in range(0,height):
    for col in range(0,width):
        if orig_img[row][col] > th:
            binary_img[row][col] = 255

cv2.imshow("Slow Binary", binary_img)


# Faster thresholded binary image
ret, binary_img_cv2 = cv2.threshold(orig_img, th, 255, cv2.THRESH_BINARY)
cv2.imshow("Binary", binary_img_cv2)



cv2.waitKey(0)
cv2.destroyAllWindows()
pass