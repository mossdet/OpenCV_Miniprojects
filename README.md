# Chapter 2
• Generate Images based on matrices, show them with cv2.imshow and save them with cv2.imwrite
• Extract r, g and b components from BGR image.
• Convert BGR to HSV and show the h,s and v components
<br />
### RGB is a color model with three dimensions – red, green, and blue – that are mixed to produce a specific color. When defining colors in these dimensions, one has to know the sequence of colors in the color spectrum, e.g. that a mix of 100% red and green produces yellow.
### HSV is a cylindrical color model that remaps the RGB primary colors into dimensions that are easier for humans to understand. Like the Munsell Color System, these dimensions are hue, saturation, and value.
<br />
• Hue specifies the angle of the color on the RGB color circle. A 0° hue results in red, 120° results in green, and 240° results in blue.
• Saturation controls the amount of color used. A color with 100% saturation will be the purest color possible, while 0% saturation yields grayscale.
• Value controls the brightness of the color. A color with 0% brightness is pure black while a color with 100% brightness has no black mixed into the color. Because this dimension is often referred to as brightness, the HSV color model is sometimes called HSB. It is important to note that the three dimensions of the HSV color model are interdependent. If the value dimension of a color is set to 0%, the amount of hue and saturation does not matter as the color will be black. Likewise, if the saturation of a color is set to 0%, the hue does not matter as there is no color used.
• Transform and BRG image to Gray scale
• Display an image with transparency
• Learn how to apply the Gaussian Blur, Dilation and Erosion functions. 
	o The gaussian blur fades alomg a given axis, use cv2.GaussianBlur
	o The dilation effect works to turn background pixels into foreground pixels. 
	o The erosion effect works to turn foreground pixels into background pixels
• Learn how to shrink, stretch and rotate an image
• Capture video and define a callback function for Mouse events.
• Create a Drawing App
<br /><br /><br />
# Chapter 3
• Apply simple thresholding to a gray scale image with the cv2.threshold function
• Apply adaptive thresholding with the cv2.adaptiveThreshold function
• Split an HSV image and plot its components. Combine thresholded components into a single image using cv2.bitwise_and()
• Find Contours on a grayscale image with cv2.findContours(), draw contours with cv2.drawContours()
• Get Contour features: Area, Perimeter, Circularity and Center
<br /><br /><br />
# Chapter 4
• Try template matching in an image
• Try face an eye detection using the pre-trained Haar Cascade Algorithm
• Detect face and eyes on the video captured by the webcam from the laptop.
<br /><br /><br />
# HFO Features:
-	Ratio of Shortest to longest Axis

