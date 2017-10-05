import unittest
import sys
import os
sys.path.append("..") #Add top level directory to python path for imports
from corn_app import filters
from PIL import Image

class TestFilters(unittest.TestCase):
    """ Tests filters.py functions
    """
    def test_apply_black_white_filter(self):
        with open('images/ear-7-512.jpg', 'rb') as image_file:
            with Image.open(image_file) as image:
                
                test_function = filters.apply_black_white_filter

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

    def test_filter_main(self):
        test_function = filters.main
        mock_args = ['filepath', 'images/ear-7-512.jpg', 'test___filters.jpg', '0']
        mock_args.append('extra_entry')

        # test too few arguments and too many arguments
        self.assertEqual(False, test_function(mock_args[0:2]))
        self.assertEqual(False, test_function(mock_args))
        del mock_args[4]

        # test invalid image path
        bad_args = mock_args[:]
        bad_args[1] = 'bad_path'
        self.assertEqual(False, test_function(bad_args))

        # Test invalid output filename
        bad_args[1] = mock_args[1]
        bad_args[2] = 'bad_out_name'
        self.assertEqual(False, test_function(bad_args))

        # Test invalid filter ID
        bad_args[2] = mock_args[2]
        bad_args[3] = 'bad filer id'
        self.assertEqual(False, test_function(bad_args))
        
        # Test correct functionality
        self.assertEqual(True, test_function(mock_args))
        os.remove('../debug/test___filters.jpg')
        print('File removed from debug folder')

if __name__ == '__main__':
    unittest.main()

