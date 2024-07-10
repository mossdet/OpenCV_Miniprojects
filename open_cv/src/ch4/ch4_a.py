import numpy as np
import cv2 as cv2
import os
import matplotlib.pyplot as plt
 
"""
Object Detection, Template Matching
"""

sep_ids_ls = [sep_idx for sep_idx, str_char in enumerate(__file__) if str_char == os.sep]
curr_path = __file__[0:sep_ids_ls[-1]] + os.sep
in_path = __file__[0:sep_ids_ls[-3]]+ os.sep + "Input_Images"+ os.sep
out_path = __file__[0:sep_ids_ls[-3]]+ os.sep + "Output_Images"+ os.sep
os.makedirs(out_path, exist_ok=True)

# Test image
img_path  = os.path.join(in_path, "template.jpg")
template_img = cv2.imread(img_path, 0)
img_path  = os.path.join(in_path, "players.jpg")
frame_img = cv2.imread(img_path, 0)

cv2.imshow("Frame", frame_img)
cv2.imshow("Template", template_img)


result_img = cv2.matchTemplate(frame_img, template_img, cv2.TM_CCOEFF_NORMED)

min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result_img)
print(max_val, max_loc)
cv2.circle(result_img, max_loc, 15, 255, 2)
cv2.imshow("Matching", result_img)


cv2.waitKey(0)
cv2.destroyAllWindows()
pass