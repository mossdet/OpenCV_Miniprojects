import numpy as np
import cv2 as cv2
import os
import matplotlib.pyplot as plt
 
"""
Object Detection, Canny Edges
"""

sep_ids_ls = [sep_idx for sep_idx, str_char in enumerate(__file__) if str_char == os.sep]
curr_path = __file__[0:sep_ids_ls[-1]] + os.sep
in_path = __file__[0:sep_ids_ls[-3]]+ os.sep + "Input_Images"+ os.sep
out_path = __file__[0:sep_ids_ls[-3]]+ os.sep + "Output_Images"+ os.sep
os.makedirs(out_path, exist_ok=True)

# Test image
test_img_path  = os.path.join(in_path, "tomatoes.jpg")
orig_img = cv2.imread(test_img_path, 1)
hsv_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2HSV)
hsv_extract_img = hsv_img[:,:,0]
th_val = np.mean(hsv_extract_img)
th_val = 25
res, th_img = cv2.threshold(hsv_extract_img, th_val, 255, cv2.THRESH_BINARY_INV)

edges_img = cv2.Canny(orig_img, threshold1=100, threshold2=200, apertureSize=3)

# Invert edges. They will be in black
edges_img_inv = 255 - edges_img

# Use Erosion filter to increase size of border
kernel = np.ones((3,3), 'uint8')
eroded_img = cv2.erode(edges_img_inv, kernel, iterations=1)

canny_th = cv2.bitwise_and(eroded_img, th_img)

cv2.imshow("hsv_extract_img", hsv_extract_img)
cv2.imshow("TH", canny_th)
cv2.imshow("Canny_Edges", edges_img_inv)
cv2.imshow("Eroded Canny_Edges", eroded_img)
cv2.imshow("Canny_TH", canny_th)

contours, hierarchy = cv2.findContours(canny_th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

objects = orig_img.copy()
for c in contours:
    area = cv2.contourArea(c)
    if area < 300:
        # This contour is around something too small for our interest
        continue
    print("Area: ", area)
    cv2.drawContours(objects, [c], -1, (255, 255, 255), 1)
    
    # Centroids
    M = cv2.moments(c)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    # Show centroid in image
    cv2.circle(objects, (cx, cy), 4, (255, 255, 0), -1)

cv2.imshow("Final draw-ver:", objects)
# cv2.imshow("Th_Tomatoes", th_img)
# cv2.imshow("Canny", edges)

cv2.waitKey(0)
cv2.destroyAllWindows()
pass