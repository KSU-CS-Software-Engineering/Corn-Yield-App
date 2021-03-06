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
import ntpath
import os
import collections

# RGB color values
GREY_SCALE_WHITE = 255
BLACK = (0,0,0)
WHITE = (255,255,255)
RED   = (0,0,255)
BLUE  = (255,0,0)

CONTOUR_COLOR = RED
BLOCK_SIZE = 71
LINE_WIDTH = 5

#HSV yellow color boundaries
LOWER_BOUND_YELLOW = [20,100,100]
UPPER_BOUND_YELLOW = [40,255,255]

METHOD_NUMBER_BEGINNING = WATERSHED_METHOD = 0
METHOD_NUMBER_ENDING    = OTSU_METHOD      = 1

# The keys are valid inputs for the count_method argument
METHODS_DICT = {
    'watershed': WATERSHED_METHOD,
    'otsu': OTSU_METHOD
}

class Features(object):

    def __init__(self, filename, count, avg_w_h_ratio):
        self.filename      = filename
        self.count         = count
        self.avg_w_h_ratio = avg_w_h_ratio

    def to_list(self):
        return [self.filename, self.count, self.avg_w_h_ratio]

    def to_feed(self, x):
        return {x: [[self.count, self.avg_w_h_ratio]]}


def mask_yellow(image):
    """Converts all image pixels not in the yellow HSV range to black

    Args:
        image (openCV Image): An open Image object.

    Returns:
        yellow_image (openCV Image): A BGR Image with yellow pixels extracted
    """

    if image is None:
        return None

    #convert image to hsv
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    '''blur hsv image so that pixels that are
    reflection of light on kernels get some yellow in them
    '''
    blur = cv2.GaussianBlur(hsv_image, (5,5), 0)

    #create numpy arrays for lower and upper bounds of yellow
    lower_yellow = np.array(LOWER_BOUND_YELLOW)
    upper_yellow = np.array(UPPER_BOUND_YELLOW)

    #turn all pixels not in yellow range. returns an hsv image
    yellow_mask = cv2.inRange(blur, lower_yellow, upper_yellow)

    '''
    erosion:
        kernel(numpy array): When cv2.erode is called, the yellow mask
        is inspected frame by frame with the dimensions of the kernel.
        If more black pixels than yellow in a frame, the yellow pixels
        are turned black and vice versa.

        Note: not to be confused with corn kernel
    '''
    kernel = np.ones((12,4), np.uint8)
    erosion = cv2.erode(yellow_mask, kernel, iterations = 1)

    #apply the eroded image to mask original image
    yellow_image = cv2.bitwise_and(image, image, mask = erosion)

    return yellow_image


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

    # Width height ratio of all contours
    w_h_ratio = 0;

    # Sum the width height ratio of all contours while drawing them.
    for (i, c) in enumerate(contours):
        x,y,w,h = cv2.boundingRect(c)
        w_h_ratio += ( w / h )
        cv2.drawContours(image, [c], -1, CONTOUR_COLOR, LINE_WIDTH)

    # pack the result image and count into a named tuple
    contour_tuple  = collections.namedtuple('contour_tuple','image avg_w_h_ratio')
    contour_result = contour_tuple(image=image, avg_w_h_ratio=w_h_ratio / len(contours) )

    return contour_result

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

def extract_features(file_path, counting_method, output_path):
    """Finds the contours of kernels on the ears of corn
    Args:
        file_path (string)      : The file_path of the image
        counting_method (string): The counting method used
        output_path (string)    : Optional method to output intermediary images to
            output path. Pass in None otherwise.
    Returns:
        Features class -- An object containg the image's features.
    """

    file  = None
    image = None

    try:
        image = cv2.imread(file_path)
        file  = ntpath.basename(file_path)
    except Exception as e:
        print(e)

    # Countour the image.
    contour_results = find_contours(mask_yellow(image))
    contoured_image = contour_results.image
    if output_path:
        export_file = os.path.join(output_path, f'contoured_{file}')
        cv2.imwrite(export_file, contoured_image)

    # Count the front facing kernels.
    count_results   = count_kernels(contoured_image, METHODS_DICT[counting_method])
    if output_path:
        export_file = os.path.join(output_path, f'{counting_method}_{file}')
        cv2.imwrite(export_file, count_results.image)

    features = Features(file, count_results.count, contour_results.avg_w_h_ratio)

    return features

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
