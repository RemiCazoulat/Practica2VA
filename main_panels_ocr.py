import os
import cv2
from matplotlib import pyplot as plt


def load_images(path):
    ''' Load all images from a given path '''
    images = []
    for filename in os.listdir(path):
        if filename.endswith('.png'):
            image = cv2.imread(os.path.join(path, filename))
            # convert to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images.append(image)
    return images


def convert_to_binary(image, thresoldType=cv2.ADAPTIVE_THRESH_MEAN_C, blockSize=9, C=5):
    ''' Convert an image to binary '''
    # convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # apply adaptive threshold inverse
    binary = cv2.adaptiveThreshold(
        gray, 255, thresoldType, cv2.THRESH_BINARY_INV, blockSize, C)
    return binary


def detect_characters(binary, min_contour_area=15, max_contour_area=500, min_aspect_ratio=0.3, max_aspect_ratio=1.1):
    ''' Detect characters in a given binary image '''
    # find contours
    contours, _ = cv2.findContours(
        binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # filter contours based on area
    contours = [cnt for cnt in contours if min_contour_area <
                cv2.contourArea(cnt) < max_contour_area]
    # filter contours based on aspect ratio
    contours = [cnt for cnt in contours if min_aspect_ratio <
                cv2.boundingRect(cnt)[2] / cv2.boundingRect(cnt)[3] < max_aspect_ratio]
    return contours


def draw_contours(image, contours):
    ''' Draw contours in a given image '''
    image_contours = image.copy()
    cv2.drawContours(image_contours, contours, -1, (0, 255, 0), 2)
    return image_contours


if __name__ == '__main__':
    images = load_images('./Práctica2_Datos_Alumnos/test_ocr_panels/')
