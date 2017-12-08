import argparse
import corn_app.contour
import corn_app.mask
import os
import cv2
import json
import sys

json_data = ""
photo_dir = ""
output_dir = ""

def apply_mask():
    for file in os.listdir(photo_dir):
        if file.endswith('.JPG'):
            image = cv2.imread(os.path.join(photo_dir, file))
            masked_image = corn_app.mask.mask_yellow(image)
            os.chdir(output_dir)
            cv2.imwrite('mask_' + file, masked_image)


def apply_contours():
    for file in os.listdir(photo_dir):
        if file.endswith('.JPG'):
            image = cv2.imread(os.path.join(photo_dir, file))
            contoured_image = corn_app.contour.find_contours(image)
            os.chdir(output_dir)
            cv2.imwrite('contours_' + file, contoured_image)


def process():
    for file in os.listdir(photo_dir):
        if file.endswith('.JPG'):
            image = cv2.imread(os.path.join(photo_dir, file))
            contoured_image = corn_app.contour.find_contours(corn_app.mask.mask_yellow(image))
            os.chdir(output_dir)
            cv2.imwrite('processed_' + file, contoured_image)


def main(args):
    if args.mask is True:
        apply_mask()
        exit(0)

    if args.contour is True:
        apply_contours()
        exit(0)

    if args.process is True:
        process()
        exit(0)


parser = argparse.ArgumentParser(description='Applies a mask and contours to pictures of corn.', prog='Corn Kernel Counter Prep Application')
parser.add_argument('--version', action='version', version='Version 0.1.0')
parser.add_argument('-m', '--mask', action='store_true', default=False, help='Applies a mask to the corn images.')
parser.add_argument('-c', '--contour', action='store_true', default=False, help='Draws contours on the corn images.')
parser.add_argument('-p', '--process', action='store_true', default=False, help='Applies a mask then draws the contours on a masked image.')
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
