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

def convert_to_black_white(image):
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

    if __debug__ : image.save('../debug/black_white_test.jpg')

    return True

def main(argv):
    """The function called when kernel_counter.py is ran

    Converts an image into a black and white format and saves the file.
    Example: python3 kernel_counter.py file/path/image file/path/output-image.
    Will likely be removed when testing framework is added

    Args:
        argv (list): The sys.argv list

    Returns:
        None    
    """
    if len(sys.argv) == 3:
        file_path    = sys.argv[1]
        output_path  = sys.argv[2]
        image        = None
        pixel_map    = None

        try:
            image = Image.open(file_path)
            pixel_map = image.load()
        except:
            print('Image could not be opened')
            sys.exit()

        print('Converting image to black and white')
        is_success = convert_to_black_white(image, pixel_map)

        if is_success:
            print('Image successfully converted')
            try:
                image.save(output_path)
            except:
                print('Image could not be saved. Check the output path file extension')

        else:
            print('Image could not be converted')

    else:
        print("An image path was not provided")
    
if __name__ == "__main__":
    # Executes if the file is ran as a script
    main(sys.argv)













