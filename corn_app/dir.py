import cv2
import numpy as np
import os

#CONTOUR_COLOR = (0, 255, 0)
#CONTOUR_THRESHOLD = 40

#filename = 'test.JPG'
#imL = cv2.imread(filename)
#im = imL
#imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
#ret,thresh = cv2.threshold(imgray,CONTOUR_THRESHOLD,255,0)
#im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#cv2.drawContours(im, contours, -1, CONTOUR_COLOR, 3)
#cv2.imwrite('x' + filename, im)

GREY_SCALE_WHITE = 255
BLACK = (0,0,0)
GREEN = (0, 255, 0)

CONTOUR_COLOR = GREEN
BLOCK_SIZE = 71
LINE_WIDTH = 5

photo_dir = '/home/tyler/corn_photos/'
output_dir = '/home/tyler/found_contours_corn_photos'

for file in os.listdir(photo_dir):
    if file.endswith('.JPG'):
        image = cv2.imread(os.path.join(photo_dir, file))

        imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thres = cv2.adaptiveThreshold(imgray, GREY_SCALE_WHITE, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,BLOCK_SIZE,0)
        im2, contours, hierarchy = cv2.findContours(thres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, CONTOUR_COLOR, LINE_WIDTH)
        os.chdir(output_dir)

        cv2.imwrite('x' + file, image)
