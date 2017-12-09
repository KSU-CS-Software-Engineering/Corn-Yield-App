"""Finds the contours of kernels on the ears of corn
Attributes:
    CONTOUR_COLOR (tuple): RGB color value used to draw the contours
    BLOCK_SIZE (int): The pixel size of the square to find a threshold for
    LINE_WIDTH (int): The width of contour lines
    COUNTING_METHODS (list(functions)): List containing all counting functions
"""
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
import cv2
import numpy as np
import collections

# RGB color values
GREY_SCALE_WHITE = 255
BLACK = (0,0,0)
WHITE = (255,255,255)
GREEN = (0,255,0)
RED   = (255,0,0)
BLUE  = (0,0,255)

CONTOUR_COLOR = BLUE
BLOCK_SIZE = 71
LINE_WIDTH = 5

METHOD_NUMBER_BEGINNING = WATERSHED_METHOD = 0
METHOD_NUMBER_ENDING    = OTSU_METHOD      = 1

def find_contours(image):
    """Finds the contours of kernels on the ears of corn
    Args:
        image (openCV Image): An open Image object.
    Returns:
        Image -- An image with the contours drawn in
    """
    if image is None:
        return None

    imgray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    thres = cv2.adaptiveThreshold(imgray, GREY_SCALE_WHITE, cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,BLOCK_SIZE,0)
    im2, contours, hierarchy = cv2.findContours(thres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, CONTOUR_COLOR, LINE_WIDTH)

    return image

def watershed_method(image):
    """Counts the kernels from a masked, contoured image of corn using
       the watershed function

    Args:
        image (openCV Image): An open Image object.
    Returns:
        Named tuple -- A tuple containing the counted image and kernel count
                       collections.namedtuple('kernel_count_tuple','image count')
    """

    # perform pyramid mean shift filtering
    # to aid the thresholding step
    shifted = cv2.pyrMeanShiftFiltering(image, 21, 51)

    # convert the mean shift image to grayscale, then apply
    # Otsu's thresholding
    gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # compute the exact Euclidean distance from every binary
    # pixel to the nearest zero pixel, then find peaks in this
    # distance map
    D = ndimage.distance_transform_edt(thresh)
    localMax = peak_local_max(D, indices=False, min_distance=20,
        labels=thresh)

    # perform a connected component analysis on the local peaks,
    # using 8-connectivity, then appy the Watershed algorithm
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=thresh)

    # loop over the unique labels returned by the Watershed
    # algorithm
    for label in np.unique(labels):
        # if the label is zero, we are examining the 'background'
        # so simply ignore it
        if label == 0:
            continue

        # otherwise, allocate memory for the label region and draw
        # it on the mask
        mask = np.zeros(gray.shape, dtype="uint8")
        mask[labels == label] = 255

        # detect contours in the mask and grab the largest one
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)[-2]
        c = max(cnts, key=cv2.contourArea)

        # draw a circle enclosing the object
        ((x, y), r) = cv2.minEnclosingCircle(c)
        cv2.circle(image, (int(x), int(y)), int(r), (0, 255, 0), 2)
        cv2.putText(image, "#{}".format(label), (int(x) - 10, int(y)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # pack the result image and count into a named tuple
    kernel_count_tuple = collections.namedtuple('kernel_count_tuple','image count')
    count_result       = kernel_count_tuple(image=image, count=len(np.unique(labels)) - 1)

    return count_result

def otsu_method(image):
    """Counts the kernels from a masked, contoured image of corn using
       Otsu thresholding.

    Args:
        image (openCV Image): An open Image object.
    Returns:
        Named tuple -- A tuple containing the counted image and kernel count
                       collections.namedtuple('kernel_count_tuple','image count')
    """

    # perform pyramid mean shift filtering
    # to aid the thresholding step
    shifted = cv2.pyrMeanShiftFiltering(image, 21, 51)

    # convert the mean shift image to grayscale, then apply
    # Otsu's thresholding
    gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # find contours in the thresholded image
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)[-2]

    # loop over the contours
    for (i, c) in enumerate(cnts):
        # draw the contour
        ((x, y), _) = cv2.minEnclosingCircle(c)
        cv2.putText(image, "#{}".format(i + 1), (int(x) - 10, int(y)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.drawContours(image, [c], -1, (0, 255, 0), 2)

    # pack the result image and count into a named tuple
    kernel_count_tuple = collections.namedtuple('kernel_count_tuple','image count')
    count_result       = kernel_count_tuple(image=image, count=len(cnts))

    return count_result

COUNTING_METHODS = [watershed_method, otsu_method]

def count_kernels(image, method_number):
    """Routes the image to the specified counting method

    Args:
        image (openCV Image): An open Image object.
        method_number (int) : A handle to a counting function
    Returns:
        Named tuple -- A tuple containing the counted image and kernel count
                       collections.namedtuple('kernel_count_tuple','image count')
    """

    if METHOD_NUMBER_BEGINNING <= method_number <= METHOD_NUMBER_ENDING:
        return COUNTING_METHODS[method_number](image)
    else:
        raise ValueError('Argument method_number is not within range')
