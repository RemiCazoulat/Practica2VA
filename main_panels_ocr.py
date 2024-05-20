import os
import cv2

def load_images(path):
    ''' Load all images from a given path '''
    images = []
    for filename in os.listdir(path):
        if filename.endswith('.jpg'):
            img = cv2.imread(os.path.join(path, filename))
            images.append(img)
    return images