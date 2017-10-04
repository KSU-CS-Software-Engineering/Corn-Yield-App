import unittest
import sys
import os
sys.path.append("..") #Add top level directory to python path for imports
from corn_app import preprocess
from PIL import Image

class TestKernelCounter(unittest.TestCase):
    """ Tests preprocess.py functions
    """

    def test_preprocess_standardize(self):
        with open('images/ear-7-512.jpg', 'rb') as image_file:
            with Image.open(image_file) as image:

                test_function = preprocess.standardize

                self.assertEqual(False, test_function(None))
                self.assertEqual(True, test_function(image))

    
    def test_preprocess_main(self):
        test_function = preprocess.main
        mock_args = ['filepath', 'images/ear-7-512.jpg', 'test___preprocess.jpg']
        mock_args.append('extra_entry')

        # test too few arguments and too many arguments
        self.assertEqual(False, test_function(mock_args[0:1]))
        self.assertEqual(False, test_function(mock_args))
        del mock_args[3] 

        # test invalid image path
        mock_args[1] = 'images/ear-7-512'
        self.assertEqual(False, test_function(mock_args))

        # test invalid output file name
        mock_args[1] += '.jpg'
        mock_args[2] = 'test___preprocess'
        self.assertEqual(False, test_function(mock_args))

        #test correct functionality
        mock_args[2] += '.jpg'
        self.assertEqual(True, test_function(mock_args))
        os.remove('../preprocessed/test___preprocess.jpg')
        print('File removed from preprocessed folder')

        #test non-existent image
        mock_args[1] = 'images/b.jpg'
        self.assertEqual(False, test_function(mock_args))


if __name__ == '__main__':
    unittest.main()

