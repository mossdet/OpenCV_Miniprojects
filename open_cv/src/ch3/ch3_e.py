import numpy as np
import cv2 as cv2
import os
import matplotlib.pyplot as plt
 
"""
Object Detection, Area, Perimeter, Circularity and Center
"""

sep_ids_ls = [sep_idx for sep_idx, str_char in enumerate(__file__) if str_char == os.sep]
curr_path = __file__[0:sep_ids_ls[-1]] + os.sep
in_path = __file__[0:sep_ids_ls[-3]]+ os.sep + "Input_Images"+ os.sep
out_path = __file__[0:sep_ids_ls[-3]]+ os.sep + "Output_Images"+ os.sep
os.makedirs(out_path, exist_ok=True)

# Test image
test_img_path  = os.path.join(in_path, "detect_blob.png")
orig_img = cv2.imread(test_img_path, 1)
gray_img = cv2.cvtColor(orig_img, cv2.COLOR_RGB2GRAY)
neighborhood_param = 115
th_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, neighborhood_param, 1)

cv2.imshow("TH_Image", th_img)

contours, hierarchy = cv2.findContours(th_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

img2 = orig_img.copy()

index = -1
thickness = 4
color = (255, 0, 255)

objects = np.zeros([orig_img.shape[0], orig_img.shape[1], 3], dtype='uint8')

for c in contours:
    
    cv2.drawContours(objects, [c], -1, color, -1)
    cv2.drawContours(objects, contours=[c], contourIdx=-1, color=color, thickness=-1) #thickness=-1, fill contour instead of drawing outline

    area = cv2.contourArea(c)
    perimeter = cv2.arcLength(c, closed=True)
    circularity = 4*np.pi*(area/(perimeter**2))

    # Centroids
    M = cv2.moments(c)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    # Show centroid in image
    cv2.circle(objects, (cx, cy), 4, (0,0,255), -1)

    # Show circularity metric in image 
    font = cv2.FONT_HERSHEY_SIMPLEX 
    org = (int(cx), cy)
    fontScale = 0.5
    thickness = 2
    cv2.putText(objects, f"{circularity:.2f}", org, font, fontScale, (255, 255, 255) , thickness, cv2.LINE_AA, bottomLeftOrigin = False) 

    metrics_str = f"Area: {area}, Perimeter: {perimeter}, Circularity: {circularity}"
    print(metrics_str)

cv2.imshow("Contours", objects)

cv2.waitKey(0)
cv2.destroyAllWindows()
pass