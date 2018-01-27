import unittest
import sys
import os
import csv
import cv2
sys.path.append("..") #Add top level directory to python path for imports
from corn_app import contour
from corn_app import mask

header = ['corn #', 'experimental front kernel', 'theoretical front kernel', 'percent error']
ERROR_MARGIN = 20 

class TestFeatures(unittest.TestCase):

    def test_otsu(self):
        image_names = sorted(os.listdir('images'))

        read_csv_file = open('absolute_features.csv', 'r')
        write_csv_file = open('test_calc_features.csv', 'w')
        feature_reader = csv.reader(read_csv_file, delimiter=',', quotechar='/', quoting=csv.QUOTE_MINIMAL)
        feature_writer = csv.writer(write_csv_file, delimiter=',', quotechar='/', quoting=csv.QUOTE_MINIMAL)

        self.assertEqual(True, len(image_names) > 0)
        corn_number = None

        # Get first row
        next(feature_reader)
        row = next(feature_reader)
        feature_writer.writerow(header)

        in_error_margin = True;

        for file in image_names:
            corn_number = int(file.split('-')[0])

            # Seek the feature reader to the current corn number
            while(corn_number != int(row[0])):
                row = next(feature_reader)
                if row is None:
                    # Consumed all rows in the absolute_features.csv file
                    # Test failed
                    read_csv_file.close()
                    write_csv_file.close()
                    self.assertTrue(False)

            image = cv2.imread(os.path.join('images', file))

            contoured_image = contour.find_contours(mask.mask_yellow(image))
            count_results   = contour.count_kernels(contoured_image, contour.OTSU_METHOD)
            front_kernel_count = count_results.count
            abs_front_kernel_count = int(row[4])

            percent_error = int(abs((front_kernel_count - abs_front_kernel_count) / abs_front_kernel_count) * 100)
            if percent_error > ERROR_MARGIN:
                in_error_margin = False

            feature_writer.writerow([file, front_kernel_count, abs_front_kernel_count, percent_error])


        read_csv_file.close()
        write_csv_file.close()

        self.assertTrue((in_error_margin))


if __name__ == '__main__':
    unittest.main()