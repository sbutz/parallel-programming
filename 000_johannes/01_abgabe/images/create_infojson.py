#!/usr/bin/env python3

import os
from PIL import Image
# suppress decompression bomb warning
Image.MAX_IMAGE_PIXELS = 120000000 

import json


def getImageDimensions(imagePath):
    with Image.open(imagePath) as img:
        width, height = img.size
        return (width, height)
    
fileInfo = []
for filename in os.listdir('.'):
    if filename.endswith('.jpg') and os.path.isfile(filename):
        width, height = getImageDimensions(filename)

        fileInfo.append({
            'filename': filename,
            'width': width,
            'height': height
        })
    
with open(os.path.join('.', 'info.json'), 'w') as file:
    json.dump(fileInfo, file, indent=4)
