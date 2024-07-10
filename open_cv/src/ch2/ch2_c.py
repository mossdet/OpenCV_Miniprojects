import numpy as np
import cv2 as cv2
import os
import matplotlib.pyplot as plt
 
"""
Read in an rgb image and transorm it to gray scale
Read in an rgb image and use one of the rgb colors as the transparency map
Show how the cv2 imshow can't show the transparency map but the matplotlib imshow can
Show how OpenCV represents RGB images as multi-dimensional NumPy arrays…but in reverse order! 
This means that in OpenCV images are actually represented in BGR order rather than RGB!
"""

sep_ids_ls = [sep_idx for sep_idx, str_char in enumerate(__file__) if str_char == os.sep]
curr_path = __file__[0:sep_ids_ls[-1]] + os.sep
in_path = __file__[0:sep_ids_ls[-3]]+ os.sep + "Input_Images"+ os.sep
out_path = __file__[0:sep_ids_ls[-3]]+ os.sep + "Output_Images"+ os.sep
os.makedirs(out_path, exist_ok=True)

# Test image
test_img_path  = os.path.join(in_path, "butterfly.jpg")
color_img = cv2.imread(test_img_path, 1)
gray_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2GRAY)
out_img_path  = out_path + "Gray_Butterfly.png"
cv2.imshow("Gray_Butterfly", gray_img)
plt.figure(num='Gray_Butterfly')
imgplot_1 = plt.imshow(gray_img, cmap='gray')
plt.title("Gray_Butterfly")
plt.show(block=False)
cv2.imwrite(out_img_path, gray_img)

b = color_img[:,:,0]
g = color_img[:,:,1]
r = color_img[:,:,2]

# use the green color as the transparency, meaning everything not green will fade out
bgra_img = cv2.merge((b,g,r,g))
rgba_img = cv2.cvtColor(bgra_img, cv2.COLOR_BGRA2RGBA)
# .jpg images do not support image transparency, so save it as png 
out_img_path  = out_path + "RGBA_Butterfly.png"
cv2.imshow("RGBA_Butterfly", rgba_img)
plt.figure(num='RGBA_Butterfly')
plt.imshow(rgba_img)
plt.title("RGBA_Butterfly")
plt.show(block=True)
cv2.imwrite(out_img_path, rgba_img)

cv2.waitKey(0)
cv2.destroyAllWindows()

pass
