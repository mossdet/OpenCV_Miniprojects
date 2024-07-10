import numpy as np
import cv2 as cv2
import os
 
"""
Extract the rgb colors from an image and display them separately

Convert an rgb image to HSV and show how one task may be impossible to do in one color space but trivial in another color space
"""

sep_ids_ls = [sep_idx for sep_idx, str_char in enumerate(__file__) if str_char == os.sep]
curr_path = __file__[0:sep_ids_ls[-1]] + os.sep
in_path = __file__[0:sep_ids_ls[-3]]+ os.sep + "Input_Images"+ os.sep
out_path = __file__[0:sep_ids_ls[-3]]+ os.sep + "Output_Images"+ os.sep
os.makedirs(out_path, exist_ok=True)

# Test image
test_img_path  = os.path.join(in_path, "butterfly.jpg")
test_img = cv2.imread(test_img_path, 1)
cv2.imshow("Butterfly", test_img)
cv2.moveWindow("Butterfly", 0, 0)
print("Butterfly Image shape: ", test_img.shape)
height, width, channels = test_img.shape


# Extract the r,g,b values from the image
b,g,r = cv2.split(test_img)

# create a matrix 3x the width of the original
rgb_split = np.empty([height, width*3, 3], 'uint8')

# assign the red colors
rgb_split[:,0:width] = cv2.merge([r,r,r])
# assign the green colors
rgb_split[:,width:2*width] = cv2.merge([g,g,g])
# assign the blue colors
rgb_split[:,2*width:3*width] = cv2.merge([b,b,b])

cv2.imshow("RGB Channels", rgb_split)
cv2.moveWindow("RGB Channels", 0, int(height))


# Extract the hsv (hue, saturation, value) values from the image
hsv_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv_img)
hsv_split = np.concatenate((h,s,v), axis=1)
cv2.imshow("HSV", hsv_img)
cv2.imshow("Split HSV", hsv_split)
cv2.moveWindow("Split HSV", 0, int(2*height))

# One task may be impossible to do in one color space but trivial in another color space

cv2.waitKey(0)
cv2.destroyAllWindows()

pass
