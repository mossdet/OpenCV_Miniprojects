import numpy as np
import cv2 as cv2
import os
import matplotlib.pyplot as plt
 
"""
For all objects in the image, segment them out, draw them on a blank canvas, and print the perimeter and area.
Only draw large obecjts (area greater than 1000px2)
"""

sep_ids_ls = [sep_idx for sep_idx, str_char in enumerate(__file__) if str_char == os.sep]
curr_path = __file__[0:sep_ids_ls[-1]] + os.sep
in_path = __file__[0:sep_ids_ls[-3]]+ os.sep + "Input_Images"+ os.sep
out_path = __file__[0:sep_ids_ls[-3]]+ os.sep + "Output_Images"+ os.sep
os.makedirs(out_path, exist_ok=True)

# Test image
test_img_path  = os.path.join(in_path, "fuzzy.png")
orig_img = cv2.imread(test_img_path, 1)

hsv_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv_img)

# Gray
gray_img = cv2.cvtColor(orig_img, cv2.COLOR_RGB2GRAY)

#Erode and dilate
#The erosion effect works to turn foreground pixels into background pixels
kernel = np.ones((5,5), 'uint8')
nr_iterations = 1
dilated_img = cv2.dilate(src=s, kernel=kernel,iterations=nr_iterations)
eroded_img = cv2.erode(src=s, kernel=kernel,iterations=nr_iterations)

# Threshold
th = np.mean(dilated_img)
ret, th_basic_img = cv2.threshold(dilated_img, th, 255, cv2.THRESH_BINARY)

neighborhood_param = 115
th_adapt_img = cv2.adaptiveThreshold(dilated_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, neighborhood_param, 1)

#cv2.imshow("HSV", hsv_img)
cv2.imshow("H from HSV", s)
#cv2.imshow("Gray", gray_img)
cv2.imshow("Dilated Image", dilated_img)
#cv2.imshow("Eroded Image", eroded_img)
cv2.imshow("Simple_Th_Image", th_basic_img)
#cv2.imshow("Adaptive_Th_Image", th_adapt_img)


contours, hierarchy = cv2.findContours(th_basic_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
objects = np.zeros([th_basic_img.shape[0], th_adapt_img.shape[1]], dtype='uint8')
for c in contours:

    color1 = (list(np.random.choice(range(256), size=3)))  
    color =[int(color1[0]), int(color1[1]), int(color1[2])] 

    area = cv2.contourArea(c)
    # if area < 1000:
    #     # This contour is around something too small for our interest
    #     continue

    print("Area: ", area)
    cv2.drawContours(objects, [c], -1, color=color, thickness=-1)
    
    # Centroids
    M = cv2.moments(c)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    # Show centroid in image
    cv2.circle(objects, (cx, cy), 4, (0, 255, 255), -1)

cv2.imshow("Final draw-ver:", objects)

cv2.waitKey(0)
cv2.destroyAllWindows()