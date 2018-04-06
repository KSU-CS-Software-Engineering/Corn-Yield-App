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

def natural_sort(l):
    """Used to sort the filenames numerically without leading zeros

    :param
        l: A list of photos
    :return:
        Returns a list of photos sorted alphanumerically
    """
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)


def process(counting_method, export_flag):
    """Processes a photo and prepares it for the TensorFlow code.

    :param
        counting_method: The counting method used to obtain the front-facing kernel count. Currently, this is fixed to the 'otsu' method.
    :param
        export_flag: A true or false flag stating if the user would like the photos exported or not.
    """
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
                    cv2.imwrite(f'{mask}_{file}', mask.mask_yellow(image))
                    cv2.imwrite(f'{contour}_{file}', contour.find_contours(image))
                    cv2.imwrite(f'{counting_method}_{file}', count_results.image)
                    
            else:
                print(f'{file} is not a supported image format')


def main(args):
    """Executes and controls the flow of the program

    :param
        args: The argparse arguments passed into the command line by the user.
    """
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


parser = argparse.ArgumentParser(description='A command line tool written in Python that processes the corn photos and prepares them for TensorFlow.', prog='Corn Kernel Counter Prep Application')
parser.add_argument('--version', action='version', version='Version 1.1.0')
parser.add_argument('-e', '--export', action='store_true', default=False, help='Exports the photos after they are processed.')
parser.add_argument('-d', '--data', action='store_true', default=False, help='Creates the data.csv file for training.')
parser.add_argument('-t', '--train', action='store_true', default=False, help='Trains model from dataset.csv file')
parser.add_argument('-f', '--full', action='store_true', default=False, help='prints counts')
parser.add_argument('-m', '--modelName', action='store', help='Used to pass in the name of the model being trained.')
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
