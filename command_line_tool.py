from corn_app import contour
from corn_app import mask
from corn_app import csv_features
from corn_app import trainer
import argparse
import collections
import os
import cv2
import json
import sys
import csv
import re

json_data = ""
photo_dir = ""
output_dir = ""

# The keys are valid inputs for the count_method argument
METHODS_DICT = {
    'watershed': contour.WATERSHED_METHOD,
    'otsu': contour.OTSU_METHOD
}

SUPPORTED_EXTS = ('.JPG', 'jpg')

"""
Used to sort filenames numerically without leading zeros
"""
def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)

def apply_mask(export_flag):
    for file in os.listdir(photo_dir):
        if file.endswith(SUPPORTED_EXTS):
            image = cv2.imread(os.path.join(photo_dir, file))

            print(f'Masking image {file}')
            masked_image = mask.mask_yellow(image)
            if export_flag is True:
                os.chdir(output_dir)
                cv2.imwrite('mask_' + file, masked_image)

        else:
            print(f'{file} is not a supported image format')

def apply_contours(export_flag):
    for file in os.listdir(photo_dir):
        if file.endswith(SUPPORTED_EXTS):
            image = cv2.imread(os.path.join(photo_dir, file))

            print(f'Contouring image {file}')
            contour_results = contour.find_contours(mask.mask_yellow(image))
            contoured_image = contour_results.image

            if export_flag is True:
                os.chdir(output_dir)
                cv2.imwrite('contours_' + file, contoured_image)

        else:
            print(f'{file} is not a supported image format')

def process(counting_method, export_flag):
    # Open csv file that the features will be written to
    with open(csv_features.FILENAME, 'w') as csvfile:
        feature_writer = csv.writer(csvfile, delimiter=csv_features.DELIM, quotechar=csv_features.QUOTECHAR, quoting=csv.QUOTE_MINIMAL)
        feature_writer.writerow(csv_features.HEADER)

        # Process ears from lowest corn id to highest
        sorted_photos = natural_sort(os.listdir(photo_dir))

        for file in sorted_photos:
            if file.endswith(SUPPORTED_EXTS):
                image = cv2.imread(os.path.join(photo_dir, file))

                print(f'Counting image {file}')

                # Countour the image.
                contour_results = contour.find_contours(mask.mask_yellow(image))
                contoured_image = contour_results.image

                # Count the front facing kernels.
                count_results   = contour.count_kernels(contoured_image, METHODS_DICT[counting_method])
                print(f'Visible kernels counted: {count_results.count}')

                feature_writer.writerow([file, count_results.count, contour_results.avg_w_h_ratio])
                
                if export_flag is True:
                    os.chdir(output_dir)
                    # Prepend to the file the counting method used for testing purposes.
                    cv2.imwrite(f'{counting_method}_{file}', count_results.image)
                    
            else:
                print(f'{file} is not a supported image format')

def main(args):
    apply_mask(args.export)
    apply_contours(args.export)
    process('otsu', args.export)

    if args.data is True:
        trainer.generate_training_set()
        exit(0)

    if args.train is True:
        trainer.train()
        exit(0)

    if args.full is True:
        count = trainer.get_count(100, 1.0)
        print(f'The predicted kernel count is: {count}\n')
        exit(0)

parser = argparse.ArgumentParser(description='Applies a mask and contours to pictures of corn.', prog='Corn Kernel Counter Prep Application')
parser.add_argument('count_method', help='Choose the counting method for the kernels: "watershed" or "otsu"')
parser.add_argument('--version', action='version', version='Version 0.1.0')
parser.add_argument('-e', '--export', action='store_true', default=False, help='Exports the photos after they are processed.')
parser.add_argument('-d', '--data', action='store_true', default=False, help='Creates the data.csv file for training.')
parser.add_argument('-t', '--train', action='store_true', default=False, help='Trains model from dataset.csv file')
parser.add_argument('-f', '--full', action='store_true', default=False, help='prints counts')
args = parser.parse_args()

try:
    json_data = json.load(open('config.json'))
except IOError:
    print('There was an error opening the \'config.json\' file, or it does not exist. Please create one in a similar structure to \'sample_config.json\'')
    sys.exit(1)

photo_dir = json_data['cornPhotoDir']
output_dir = json_data['contourPhotoDir']

if __name__ == '__main__':
    main(args)
