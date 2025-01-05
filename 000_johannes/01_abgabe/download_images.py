#!/usr/bin/env python3

import os
import requests
from PIL import Image
# suppress decompression bomb warning
Image.MAX_IMAGE_PIXELS = 120000000 

import json

downloadDirectory = os.path.join('.', 'images_from_web')
os.makedirs(downloadDirectory, exist_ok=True)

urls = [
    'https://upload.wikimedia.org/wikipedia/commons/thumb/4/4b/Catedral_de_Toledo.Altar_Mayor_%28huge%29.jpg/320px-Catedral_de_Toledo.Altar_Mayor_%28huge%29.jpg',
    'https://upload.wikimedia.org/wikipedia/commons/thumb/4/4b/Catedral_de_Toledo.Altar_Mayor_%28huge%29.jpg/640px-Catedral_de_Toledo.Altar_Mayor_%28huge%29.jpg',
    'https://upload.wikimedia.org/wikipedia/commons/thumb/4/4b/Catedral_de_Toledo.Altar_Mayor_%28huge%29.jpg/800px-Catedral_de_Toledo.Altar_Mayor_%28huge%29.jpg',
    'https://upload.wikimedia.org/wikipedia/commons/thumb/4/4b/Catedral_de_Toledo.Altar_Mayor_%28huge%29.jpg/1024px-Catedral_de_Toledo.Altar_Mayor_%28huge%29.jpg',
    'https://upload.wikimedia.org/wikipedia/commons/thumb/4/4b/Catedral_de_Toledo.Altar_Mayor_%28huge%29.jpg/1280px-Catedral_de_Toledo.Altar_Mayor_%28huge%29.jpg',
    'https://upload.wikimedia.org/wikipedia/commons/thumb/4/4b/Catedral_de_Toledo.Altar_Mayor_%28huge%29.jpg/2560px-Catedral_de_Toledo.Altar_Mayor_%28huge%29.jpg',
    'https://upload.wikimedia.org/wikipedia/commons/4/4b/Catedral_de_Toledo.Altar_Mayor_%28huge%29.jpg'
]

def getImageDimensions(imagePath):
    with Image.open(imagePath) as img:
        width, height = img.size
        return (width, height)
    
fileInfo = []
for i, url in enumerate(urls, start=1):
    headers = {'User-Agent': 'download_images.py/0.1 (jaybee@sihlfall.ch)'}
    response = requests.get(url, headers=headers)
    response.raise_for_status()

    filename = f'toledo{i:02d}.jpg'
    filePath = os.path.join(downloadDirectory, filename)
    with open(filePath, 'wb') as file:
        file.write(response.content)

    width, height = getImageDimensions(filePath)

    fileInfo.append({
        'url': url,
        'filename': filename,
        'width': width,
        'height': height
    })
    
with open(os.path.join(downloadDirectory, 'info.json'), 'w') as file:
    json.dump(fileInfo, file, indent=4)
