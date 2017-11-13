import json
import os
import contour
import cv2

json_data = json.load(open('../config.json'))

photo_dir = json_data['cornPhotoDir']
output_dir = json_data['contourPhotoDir']

for file in os.listdir(photo_dir):
    if file.endswith('.JPG'):
        image = cv2.imread(os.path.join(photo_dir, file))
        contoured_image = contour.find_contours(image)
        os.chdir(output_dir)
        cv2.imwrite('contours_' + file, contoured_image)
