"""
filters.py
A set of image filters used in the project
"""
from PIL import Image
import sys
import os

#A filter function identifier used in the main function
BLACK_WHITE_FILTER_ID = '0'

# Coefficients used to calculate the luminance of a color channel.
# Based from the ITU BT.709 standard
LUMA_RED   = 0.2126
LUMA_GREEN = 0.7152
LUMA_BLUE  = 0.0722

# The threshold determing if a pixel should be black or white based from the lumanince
LUMA_THRESHOLD = 150

def apply_black_white_filter(image):
    """Converts an image into black and white based from its lumanince.

    Currently only supports RGB image modes.
    
    Args:
        image (Image): An open Image instance.

    Returns:
        bool: True if the image is converted, false otherwise.
    """
    if not image or image.mode != 'RGB':
        return False

    pixel_map = image.load()

    image_width, image_height = image.size

    for i in range(image_width):
        for j in range(image_height):
            new_pixel_value = (0, 0, 0)
            pixel = pixel_map[i, j]

            # Calculate the total luminance of a pixel.
            luma = pixel[0] * LUMA_RED + pixel[1] * LUMA_GREEN + pixel[2] * LUMA_BLUE
            
            # Set pixel to white if luma is larger than LUMA_THRESHOLD
            if luma > LUMA_THRESHOLD: new_pixel_value = (255, 255, 255)

            pixel_map[i, j] = new_pixel_value

    return True

def main(argv):
    """The function called when kernel_counter.py is ran.

    Converts an image into a black and white format and saves the file
    to the debug folder as black_white_test.jgp. This allows personal testing
    of any filters without affecting any other files.
    The argv needs a image path, output file name, and a filer number.

    Example:
        python3 kernel_counter.py /path/to/image.jpg test1.jpg 1

    Args:
        argv (list): The sys.argv list.

    Returns:
        bool: True if the image was saved, False if it was not. 
    """
    if len(argv) == 4:
        file_path     = argv[1]
        file_name     = argv[2]
        filter_id     = argv[3]
        image         = None

        try:
            with open(file_path, 'rb') as image_file:
                with Image.open(image_file) as image:
                    is_success = False

                    # Determine which filter to use
                    if filter_id == BLACK_WHITE_FILTER_ID:
                        print('Converting image to black and white')
                        is_success = apply_black_white_filter(image)
                    else:
                        print(f'{filter_id} is not a valid filter ID')

                    if is_success:
                        try:
                            if not os.path.isdir('../debug'): 
                                print( 'Creating debug folder')
                                curfilePath = os.path.abspath(__file__)
                                curDir = os.path.abspath(os.path.join(curfilePath,os.pardir))
                                parentDir = os.path.abspath(os.path.join(curDir,os.pardir))
                                os.makedirs(os.path.join(parentDir, 'debug'))
                                
                            image.save(f'../debug/{file_name}')
                            print('File saved to debug folder')
                            return True
                        except:
                            print('Image could not be saved.')
                            return False
                    else:
                        print('Image could not be converted')
                        return False
        except:
            print('Image could not be opened')
            return False
    else:
        print('File requires three arguments')
        return False
    
if __name__ == "__main__":
    # Executes if the file is ran as a script
    main(sys.argv)
