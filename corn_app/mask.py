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