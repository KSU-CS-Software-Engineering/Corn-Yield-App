"""
preprocess.py
Resizes images to a standard size
"""
from PIL import Image
import sys
import os

STANDARD_WIDTH = 1280 # This is our standard image width
STANDARD_HEIGHT = 720 # This is our standard image height

def standardize(image):
    """
    Standardizes image parameters.
    For size parameters, we are standardizing to a resolution of 1280 x 720

    Args:
        image (Image): An open Image instance.

    Returns:
        image(Image): new Image instance with new size parameters
    """
    if not image:
        print('No Image Found')
        return None

    print('Resizing Image...')

    image = image.resize((STANDARD_WIDTH, STANDARD_HEIGHT))
    return image

def main(argv):
    """
        Resizes an image.
        The argv needs an image path and an output file name

        Example:
            python preprocess.py /path/to/image.jpg preprocess.jpg

        Args:
            argv (list): The sys.argv list

        Returns:
            bool: True if image was saved, False otherwise.
    """

    if len(argv) != 3:
        print('File requires three arguments')
        return False

    file_path = argv[1]
    file_name = argv[2]
    image = None

    try:
        with open(file_path, 'rb') as image_file:
            with Image.open(image_file) as image:
                try:
                    image = standardize(image)

                    if image.width != STANDARD_WIDTH and image.height != STANDARD_HEIGHT:
                        print('Image was not resized')
                        return False

                    print('Image Resized')

                    try:
                        if not os.path.exists('../preprocessed/'):
                            # Create preprocessed folder if it does not exist
                            os.makedirs('../preprocessed/')

                        image.save(f'../preprocessed/{file_name}')
                        print('Image Saved to preprocessed folder')
                        return True

                    except:
                        print('Image could not be saved')
                        return False

                except:
                    print('Image was not resized.')
                    return False
    except:
        print('Image could not be opened')
        return False

if __name__ == "__main__":
    # Executes if the file is ran as a script
    main(sys.argv)