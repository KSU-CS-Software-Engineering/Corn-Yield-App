import unittest
import sys
sys.path.append("..") #Add top level directory to python path for imports
from corn_app import kernel_counter
from PIL import Image

class TestKernelCounter(unittest.TestCase):
    """
    Tests kernel_counter.py functions
    """
    def test_convert_to_black_white(self):
        # Initialize local variables
        with open('images/ear-7-512.jpg', 'rb') as image_file:
            with Image.open(image_file) as image:
                
                test_function = kernel_counter.convert_to_black_white

                self.assertEqual(False, test_function(None))
                self.assertEqual(True, test_function(image))

                color_pixel_found = False
                white_black_pixels = [(255,255,255),(0,0,0)]
                pixel_map = image.load()

                for i in range(image.width):
                    for j in range(image.height):
                        pixel = pixel_map[i, j]
                        if pixel not in white_black_pixels:
                            color_pixel_found = True
                            print(f'Color found at i: {i} j: {j} pixel:{pixel}')
                            break

                self.assertEqual(False, color_pixel_found)


if __name__ == '__main__':
    unittest.main()


