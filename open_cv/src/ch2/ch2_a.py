import numpy as np
import cv2 as cv2
import os
 
"""
Read and write images with cv2

Generate an all black, all ones, all white and all blue image.

"""

sep_ids_ls = [sep_idx for sep_idx, str_char in enumerate(__file__) if str_char == os.sep]
curr_path = __file__[0:sep_ids_ls[-1]] + os.sep
in_path = __file__[0:sep_ids_ls[-3]]+ os.sep + "Input_Images"+ os.sep
out_path = __file__[0:sep_ids_ls[-3]]+ os.sep + "Output_Images"+ os.sep
os.makedirs(out_path, exist_ok=True)

# Test image
test_img_path  = os.path.join(in_path, "opencv-logo.png")
test_img = cv2.imread(test_img_path,1)
cv2.imshow("Test Image", test_img)
print(test_img[:,:,0])
out_img_path  = out_path + "Test_Image_Out.png"
cv2.imwrite(out_img_path, test_img)

# Black image
black_img = np.zeros([115,200,1], 'uint8')
cv2.imshow("Black", black_img)
print(black_img[:,:,0])
out_img_path  = out_path + "Black.png"
cv2.imwrite(out_img_path, black_img)

# All ones image
ones_img = np.ones([150,200,3], 'uint8')
cv2.imshow('Ones', ones_img)
print(ones_img[0,0,:])
out_img_path  = out_path + "Ones.png"
cv2.imwrite(out_img_path, ones_img)

# White image
white_img = np.ones([150,200,3], 'uint8')
white_img *= (2**8)-1
cv2.imshow('White', white_img)
print(white_img[0,0,:])
out_img_path  = out_path + "White.png"
cv2.imwrite(out_img_path, white_img)

# Blue image
blue_img = white_img.copy()
blue_img[:,:,:] = (255, 0, 0)
cv2.imshow('Blue', blue_img)
print(blue_img[0,0,:])
out_img_path  = out_path + "Blue.png"
cv2.imwrite(out_img_path, blue_img)

cv2.waitKey(0)
cv2.destroyAllWindows()


