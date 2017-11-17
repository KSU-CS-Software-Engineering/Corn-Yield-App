""" Converts all image pixels not in the required HSV range to black

Attributes:
	LOWER_BOUND (array): HSV values for the lower boundary of color
	UPPER_BOUND (array): HSV values for the upper boundary of color
"""
import cv2
import numpy as np

#HSV yellow color boundaries
LOWER_BOUND_YELLOW = [20,100,100]
UPPER_BOUND_YELLOW = [40,255,255]

def mask_yellow(image):
	"""Converts all image pixels not in the yellow HSV range to black

	Args:
		image (openCV Image): An open Image object.

	Returns:
		bgr_yellow_image (openCV Image): An Image with yellow pixels extracted
	"""

	if image is None:
		return None

	hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	blur = cv2.GaussianBlur(hsv_image, (5,5), 0)

	lower_yellow = np.array(LOWER_BOUND_YELLOW)
	upper_yellow = np.array(UPPER_BOUND_YELLOW)

	yellow_mask = cv2.inRange(blur, lower_yellow, upper_yellow)

	hsv_yellow_image = cv2.bitwise_and(image, image, mask = yellow_mask)
	bgr_yellow_image = cv2.cvtColor(hsv_yellow_image, cv2.COLOR_HSV2BGR)

	return bgr_yellow_image