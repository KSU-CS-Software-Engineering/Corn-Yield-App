import json
import os
import contour

json_data = json.load(open('../config.json'))

photo_dir = json_data['cornPhotoDir']
output_dir = json_data['contourPhotoDir']

for file in os.listdir(photo_dir):
    if file.endswith('.JPG'):
        contour.find_contours(file)
