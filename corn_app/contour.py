"""Finds the contours of kernels on the ears of corn

Attributes:
    CONTOUR_COLOR (tuple): RGB color value used to draw the contours
    BLOCK_SIZE (int): The pixel size of the square to find a threshold for
    LINE_WIDTH (int): The width of contour lines
"""
import cv2
import numpy as np

# RGB color values
GREY_SCALE_WHITE = 255
BLACK = (0,0,0)
GREEN = (0,255,0)

CONTOUR_COLOR = GREEN
BLOCK_SIZE = 71
LINE_WIDTH = 5

#HSV yellow color boundaries
LOWER = [20,70,70]
UPPER = [40,255,255]

def find_contours(image):
    """Finds the contours of kernels on the ears of corn

    Args:
        image (openCV Image): An open Image object.

    Returns:
        yellow_hsv(Image) -- An image with the contours drawn in
        contours(list): List of contours drawn
    """
    if image is None:
        return None

    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_yellow = np.array(LOWER)
    upper_yellow = np.array(UPPER)

    yellow_mask = cv2.inRange(image_hsv, lower_yellow, upper_yellow)
    yellow_hsv = cv2.bitwise_and(image, image, mask = yellow_mask)
    yellow_bgr = cv2.cvtColor(yellow_hsv, cv2.COLOR_HSV2BGR)

    imgray = cv2.cvtColor(yellow_bgr,cv2.COLOR_BGR2GRAY)
    thres = cv2.adaptiveThreshold(imgray, GREY_SCALE_WHITE, cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,BLOCK_SIZE,0)
    im2, contours, hierarchy = cv2.findContours(thres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(yellow_hsv, contours, -1, CONTOUR_COLOR, LINE_WIDTH)

    return yellow_hsv, contours

