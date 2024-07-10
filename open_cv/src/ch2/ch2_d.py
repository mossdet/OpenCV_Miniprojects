import numpy as np
import cv2 as cv2
import os
import matplotlib.pyplot as plt
 
"""
Gaussian Blur, Dilation, Erosion

"""

sep_ids_ls = [sep_idx for sep_idx, str_char in enumerate(__file__) if str_char == os.sep]
curr_path = __file__[0:sep_ids_ls[-1]] + os.sep
in_path = __file__[0:sep_ids_ls[-3]]+ os.sep + "Input_Images"+ os.sep
out_path = __file__[0:sep_ids_ls[-3]]+ os.sep + "Output_Images"+ os.sep
os.makedirs(out_path, exist_ok=True)

# Test image
test_img_path  = os.path.join(in_path, "thresh.jpg")
orig_img = cv2.imread(test_img_path, 1)
cv2.imshow("Original Image", orig_img)


"""
Gaussian Blur
"""
# (5, 55) will blurr a lot more along the y axis than the x axis
blur_img1 = cv2.GaussianBlur(orig_img, (5, 55), 0)
cv2.imshow("Gaussian Blurred Image 1", blur_img1)

# (5, 55) will blurr a lot more along the y axis than the x axis
blur_img2 = cv2.GaussianBlur(orig_img, (55, 5), 0)
cv2.imshow("Gaussian Blurred Image 2", blur_img2)


"""
Dilationand Erosion
Help expand or contract the foreground pixels of an image to help remove or accentuate small pixel details such as speckles.
They work by sliding a kernel template, a small square, across an image.
The dilation effect works to turn background pixels into foreground pixels
The erosion effect works to turn foreground pixels into background pixels
"""
kernel = np.ones((5,5), 'uint8')
nr_iterations = 1
dilated_img = cv2.dilate(src=orig_img, kernel=kernel,iterations=nr_iterations)
eroded_img = cv2.erode(src=orig_img, kernel=kernel,iterations=nr_iterations)
cv2.imshow("Dilated Image", dilated_img)
cv2.imshow("Eroded Image", eroded_img)

cv2.waitKey(0)
cv2.destroyAllWindows()

pass
