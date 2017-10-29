"""Finds the contours of an image

The COUNTOUR_THRESHOLD will be found from an image's
luminosity, allowing this module to work for varyiny
lighting conditions in the future.

Module is currenlty in the experimental phase

Attributes:
    CONTOUR_COLOR (tuple): RGB color value used to draw the contours
    COUTOUR_THRESHOLD (int): The boundry between turning a pixel white or black
"""

import cv2
import numpy as np

# Green RGB  value
CONTOUR_COLOR = (0, 255, 0) 
COUTOUR_THRESHOLD = 40

#image name is currently hard coded in
filename = 'test4.JPG'
imL = cv2.imread(filename)
im = imL
imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(imgray,COUTOUR_THRESHOLD,255,0)
im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(im, contours, -1, CONTOUR_COLOR, 3)
cv2.imwrite('x' + filename, im)
