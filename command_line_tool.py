from corn_app import contour
from corn_app import mask
import argparse
import collections
import os
import cv2
import json
import sys

json_data = ""
photo_dir = ""
output_dir = ""

SUPPORTED_EXTS = ('.JPG', 'jpg')

def apply_mask():
    for file in os.listdir(photo_dir):
        if file.endswith(SUPPORTED_EXTS):
            image = cv2.imread(os.path.join(photo_dir, file))

            print(f'Masking image {file}')
            masked_image = mask.mask_yellow(image)

            os.chdir(output_dir)
            cv2.imwrite('mask_' + file, masked_image)
        else:
            print(f'{file} is not a supported image format')

def apply_contours():
    for file in os.listdir(photo_dir):
        if file.endswith(SUPPORTED_EXTS):
            image = cv2.imread(os.path.join(photo_dir, file))

            print(f'Contouring image {file}')
            contoured_image = contour.find_contours(mask.mask_yellow(image))

            os.chdir(output_dir)
            cv2.imwrite('contours_' + file, contoured_image)
        else:
            print(f'{file} is not a supported image format')

def process():
    for file in os.listdir(photo_dir):
        if file.endswith(SUPPORTED_EXTS):
            image = cv2.imread(os.path.join(photo_dir, file))

            print(f'Counting image {file}')
            contoured_image = contour.find_contours(mask.mask_yellow(image))
            count_results   = contour.count_kernels(contoured_image, contour.OTSU_METHOD)
            print(f'Visible kernels counted: {count_results.count}')

            os.chdir(output_dir)
            cv2.imwrite('processed_' + file, count_results.image)
        else:
            print(f'{file} is not a supported image format')

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
