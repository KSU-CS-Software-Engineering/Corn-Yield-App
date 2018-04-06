import unittest
import sys
import os
import csv
import cv2
sys.path.append("..") #Add top level directory to python path for imports
from corn_app import feature
from corn_app import trainer
import re

header = "Image file | Calculated final count | Absoulute final Count | Percent Error".split('|')
ERROR_MARGIN = 20 
TEST_MODEL     = 'fullbatch'
TEST_ITERATION = 1000

"""
Used to sort filenames numerically without leading zeros
"""
def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)

class TestFeatures(unittest.TestCase):


    def test_count(self):
        os.chdir('../')

        # List of sorted file names excluding hidden files.
        image_names = natural_sort(filter( lambda f: not f.startswith('.'), os.listdir('tests/images')))

        write_csv_file   = open('tests/test_calc_final_count.csv', 'w')
        total_count_file = open('csv/total_kernel_counts.csv', 'r')

        total_count_reader = csv.reader(total_count_file , delimiter=',', quotechar='/', quoting=csv.QUOTE_MINIMAL)
        feature_writer     = csv.writer(write_csv_file, delimiter=',', quotechar='/', quoting=csv.QUOTE_MINIMAL)

        self.assertEqual(True, len(image_names) > 0)
        corn_number = None

        # Call next() to skip past header row.
        next(total_count_reader)

        # Get first data row of csv file.
        total_count_row = next(total_count_reader)

        in_error_margin = True;

        feature_writer.writerow(header)

        for file in image_names:
            print(f"Counting {file}")
            corn_number   = int(file.split('_')[0])
            corn_total_id = int(total_count_row[0])

            # Seek the absolute reader to the current corn feature id
            while corn_total_id < corn_number:
                try:
                    total_count_row = next(total_count_reader)
                    corn_total_id   = int(total_count_row[0])
                except StopIteration:
                    # Consumed all rows in the absolute_features.csv file.
                    # This means we're training a corn id we do not have a final count
                    # for. The program will be terminated.
                    print(f"Final kernel count does not exist for corn ID: {corn_number}.")
                    print("Test will now terminate.")
                    write_csv_file.close()
                    total_count_file.close()
                    self.assertTrue(False)

            file_path       = os.path.join('tests/images', file)
            features        = feature.extract_features(file_path, 'otsu', None)
            final_count     = trainer.get_count(TEST_MODEL, features)
            abs_final_count = int(total_count_row[3])

            percent_error = int(abs((final_count - abs_final_count) / abs_final_count) * 100)

            if percent_error > ERROR_MARGIN:
                in_error_margin = False

            feature_writer.writerow([file, final_count, abs_final_count, percent_error])

        write_csv_file.close()
        total_count_file.close()

        self.assertTrue((in_error_margin))


if __name__ == '__main__':
    unittest.main()