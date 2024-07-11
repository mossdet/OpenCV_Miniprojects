import numpy as np
import cv2 as cv2
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
 
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
hfo_spect_img = cv2.imread(test_img_path, 1)
hfo_spect_img = cv2.resize(hfo_spect_img, (1024, 768), interpolation=cv2.INTER_LINEAR)
height, width = hfo_spect_img.shape[0:2]

# get HSV image and components
hsv_img = cv2.cvtColor(hfo_spect_img, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv_img)

hsv_img_extract = h
hsv_img_extract_inv = 255-h

# Threshold hsv channel
th_val = np.percentile(hsv_img_extract_inv, 99, axis=None)
th_val = np.median(hsv_img_extract_inv)+5*np.std(hsv_img_extract_inv)
#th_val = np.percentile(hsv_img_extract_inv,99.1)
ret, th_img = cv2.threshold(hsv_img_extract_inv, th_val, 255, cv2.THRESH_BINARY)


cv2.imshow("Original_Image", hfo_spect_img)
cv2.imshow("HSV_Extract_Image_Inv", hsv_img_extract_inv)
cv2.imshow("TH_Image", th_img)


# Get Contours
contours, hierarchy = cv2.findContours(th_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Plot Contours
objects = hfo_spect_img.copy()
thickness = 4
color = (255, 255, 255)
maxLevel = 0
for cont_idx, contour in enumerate(contours):

    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, closed=True)
    if area < 200 or perimeter == 0:
        # This contour is around something too small for our interest
        continue
        pass
    
    circularity = 4*np.pi*(area/(perimeter**2))

    hspread = np.max(contour[:,:,0])-np.min(contour[:,:,0])
    vspread = np.max(contour[:,:,1])-np.min(contour[:,:,1])
    minAx_maxAx_ratio = 100*(np.min([hspread,vspread])/np.max([hspread,vspread]))

    # Centroids
    M = cv2.moments(contour)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    # Show contour and centroid in image
    cv2.drawContours(image=objects, contours=[contour], contourIdx=-1, color=color, thickness=thickness, hierarchy=hierarchy, maxLevel=maxLevel)
    cv2.circle(objects, (cx, cy), 4, (0,0,255), -1)

    # Show circularity metric in image 
    font = cv2.FONT_HERSHEY_SIMPLEX 
    org = (int(cx), cy)
    fontScale = 0.5
    thickness = 2

    annot_str = f"A:{area:.0f}, C:{100*circularity:.0f}, HS:{hspread:.0f}, VS:{vspread:.0f}, HVR:{minAx_maxAx_ratio:.0f}"
    cv2.putText(objects, annot_str, org, font, fontScale, (255, 255, 255) , thickness, cv2.LINE_AA, bottomLeftOrigin = False) 

    metrics_str = f"Area: {area}, Perimeter: {perimeter}, Circularity: {circularity}"
    print(metrics_str)

cv2.imshow("Final draw-ver:", objects)

cv2.waitKey(0)
cv2.destroyAllWindows()

pass