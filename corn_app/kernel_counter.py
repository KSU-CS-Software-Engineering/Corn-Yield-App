"""
kernel_counter.py
TODO: Add module description and add kernel counting function
"""
from PIL import Image
import sys

# Coefficients used to calculate the luminance of a color channel.
# Based from the ITU BT.709 standard
LUMA_RED   = 0.2126
LUMA_GREEN = 0.7152
LUMA_BLUE  = 0.0722

# The threshold determing if a pixel should be black or white based from the lumanince
LUMA_THRESHOLD = 150

def convertToBlackWhite(image, pixelMap):
"""Converts an image into black and white based from its lumanince.

This function was ported and improved from the 2016 Corn Yield Senior design project.

Args:
    image (Image): An open Image instance.
    pixelMap (PixelAccess): A buffer containing the bitmap data.

Returns:
    bool: True if the image is converted, false otherwise.
"""
    # RGB is currently only supported.
    if image.mode != 'RGB':
        return False

    imageWidth, imageHeight = image.size

    for i in range(imageWidth):
        for j in range(imageHeight):
            newPixelValue = (0, 0, 0)
            pixel = pixelMap[i, j]

            # Calculate the total luminance of a pixel.
            luma = pixel[0] * LUMA_RED + pixel[1] * LUMA_GREEN + pixel[2] * LUMA_BLUE
 
            if luma > LUMA_THRESHOLD: newPixelValue = (255, 255, 255)

            pixelMap[i, j] = newPixelValue

    return True
    
if __name__ == "__main__":
    # Execute only if the file is ran as a script.
    # Example: python3 ip.py file/path/image file/path/output-image.
    if len(sys.argv) == 3:
        filePath     = sys.argv[1]
        outPutPath   = sys.argv[2]
        image        = None
        pixelMap     = None

        try:
            image = Image.open(filePath)
            pixelMap = image.load()
        except:
            print('Image could not be opened')
            sys.exit()

        print('Converting image to black and white')
        isSuccess = convertToBlackWhite(image, pixelMap)

        if isSuccess:
            print('Image successfully converted')
            image.save(outPutPath)
        else:
            print('Image could not be converted')

    else:
        print("An image path was not provided")













