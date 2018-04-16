from corn_app import feature
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
import time

json_data = ""
photo_dir = ""
output_dir = ""

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


def features_process(output_path):
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

        # Process ears from lowest corn id to highest.
        sorted_photos     = natural_sort(os.listdir(photo_dir))

        # Gather progress data for the user.
        file_count        = len(sorted_photos)
        current_file_num  = 0
        start_time        = time.time();

        print('Begin processing images')
        for file in sorted_photos:
            current_file_num += 1
            if file.endswith(SUPPORTED_EXTS):
                # Print current file and current file position of all files to console.
                print('{0:30}   {1}/{2}'.format(file[0:20], current_file_num, file_count ))

                # Extract features from image and write to feature file.
                file_path = os.path.join(photo_dir, file)
                features  = feature.extract_features(file_path, 'otsu', output_path)
                feature_writer.writerow(features.to_list())
            else:
                print(f'{file} is not a supported image format')

        # Display time taken for processing.
        elapsed_time = time.time() - start_time
        m, s = divmod(elapsed_time, 60)
        h, m = divmod(m, 60)
        print ("{0:d}h:{1:d}m:{2:.2f}s".format(int(h),int(m),s))

def main(args):
    """Executes and controls the flow of the program

    :param
        args: The argparse arguments passed into the command line by the user.
    """
    output_path = None

    if args.export is True:
        output_path = output_dir

    if args.all is True:
        if args.modelname is None:
            print('A name for the new model is needed.')
            exit(0)

        features_process(output_path)
        trainer.generate_training_set(args.modelname)
        trainer.train(args.modelname)
        print(f'Model {args.modelname} trained.')
        exit(0)

    if args.features is True:
        features_process(output_path)

    if args.data is True:
        if args.modelname is None:
            print('A model name is needed for the data set.')
            exit(0)

        trainer.generate_training_set(args.modelname)
        exit(0)

    if args.train is True:
        if args.modelname is None:
            print('A name for the new model is needed.')
            exit(0)

        trainer.train(args.modelname)
        print(f'Model {args.modelname} trained.')
        exit(0)

    if args.count is True:
        if args.path is None:
            print('A file path is needed to count an image.')
            exit(0)

        if args.modelname is None:
            print('A name is needed for the model to use.')
            exit(0)

        print('Processing image.')
        features = feature.extract_features(args.path, 'otsu', output_path)
        count    = trainer.get_count(args.modelname, features)
        print(f'The predicted kernel count is: {count}\n')
        exit(0)


parser = argparse.ArgumentParser(description='A command line tool written in Python that processes the corn photos and prepares them for TensorFlow.', prog='Corn Kernel Counter Prep Application')
parser.add_argument('--version', action='version', version='Version 1.1.0')
parser.add_argument('-e', '--export',   action='store_true', default=False, help='Exports the photos after they are processed.')
parser.add_argument('-a', '--all',      action='store_true', help='Processes photos, creates the data set for a model and trains said model on the data set.')
parser.add_argument('-d', '--data',     action='store_true', default=False, help='Creates the data.csv file for training.')
parser.add_argument('-t', '--train',    action='store_true', default=False, help='Trains model from dataset.csv file')
parser.add_argument('-c', '--count',    action='store_true', default=False, help='gets_count')
parser.add_argument('-f', '--features', action='store_true', default=False, help='Applies a mask then draws the contours on a masked image.')
parser.add_argument('-m', '--modelname', action='store', help='Used to pass in the name of the model being trained.')
parser.add_argument('-p', '--path',      action='store', help='File path to an image.')

args = parser.parse_args()

try:
    json_data = json.load(open('config.json'))
except IOError:
    print('There was an error opening the \'config.json\' file, or it does not exist. Please create one in a similar structure to \'sample_config.json\'')
    sys.exit(1)

photo_dir  = json_data['cornPhotoDir']
output_dir = json_data['contourPhotoDir']

if __name__ == '__main__':
    main(args)
