import numpy as np
import cv2 as cv2
import os
import matplotlib.pyplot as plt
 
"""
Create a drawing app
"""

# Global variables
canvas = np.ones([500, 500, 3], dtype='uint8')*255 # white canvas
draw_enabled = False
drawing_color = (0, 255, 0)

# if -1, the circle would be filled, instead of having a line of thickness
line_width = -1
bttn_width = 25

r1_start_point = (0,0)
r1_end_point = (bttn_width,bttn_width)
r1_color = (0, 255, 0)
r2_start_point = (bttn_width+1, 0)
r2_end_point = (bttn_width*2+1,bttn_width)
r2_color = (255, 255, 0)

circ_radius = 10

# Callback function
def click(event, x, y, flags, param):
    global canvas
    global draw_enabled
    global drawing_color

    if event == cv2.EVENT_LBUTTONDOWN:
        col1_chosen = x >= r1_start_point[0] and x <= r1_end_point[0] and y >= r1_start_point[1] and y <= r1_end_point[1]
        col2_chosen = x >= r2_start_point[0] and x <= r2_end_point[0] and y >= r2_start_point[1] and y <= r2_end_point[1]

        if col1_chosen:
            drawing_color = r1_color
        elif col2_chosen:
            drawing_color = r2_color
        else:
            draw_enabled = True

        #print("Pressed", x,y)

    if event == cv2.EVENT_LBUTTONUP:
        draw_enabled = False

    if draw_enabled:
        #canvas[y,x,:] = drawing_color
        cv2.circle(canvas, (x,y), circ_radius, drawing_color, -1)


# Be sure that the window name here is the same as in the while loop
cv2.namedWindow("Canvas")
cv2.setMouseCallback("Canvas", click)

while(True):

    cv2.rectangle(canvas, r1_start_point, r1_end_point, r1_color, line_width)
    cv2.rectangle(canvas, r2_start_point, r2_end_point, r2_color, line_width)

    cv2.imshow("Canvas", canvas)

    ch = cv2.waitKey(1)
    if ch & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()