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


def get_contours(binary, min_contour_area=50, max_contour_area=500, min_aspect_ratio=0.3, max_aspect_ratio=1.1):
    ''' Get contours in a given binary image '''
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


def get_centroid(contour):
    ''' Get the centroid of a given contour '''
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return (0, 0)
    x = int(M["m10"] / M["m00"])
    y = int(M["m01"] / M["m00"])
    return (x, y)


def get_line(point1, point2):
    ''' Get a line passing through two points in a slope-intercept form '''
    x1, y1 = point1
    x2, y2 = point2
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1
    return (slope, intercept)


def distance_point_to_line(point, line):
    ''' Calculate the distance between a point and a line '''
    slope, intercept = line
    x, y = point
    return abs(slope * x - y + intercept) / (slope**2 + 1)**0.5


def get_lines(contours, threshold=15):
    ''' Get lines in a given image in a slope-intercept form'''
    # calculate centroids of each contour
    points = [get_centroid(cnt) for cnt in contours]
    # remove duplicates
    points = list(set(points))
    # sort the centroids based on the y-coordinate
    points = sorted(points, key=lambda x: x[1])
    # get a line passing through the first two centroids and add it to the list of lines
    line = get_line(points[0], points[1])
    lines = [line]
    for i in range(2, len(points)-1):
        # calculate the distance between the last added line and the centroid of the next contour
        # if the distance is less than a threshold, move to the next centroid
        if distance_point_to_line(points[i], line) < threshold:
            continue
        # otherwise, get another line passing between the last two centroids
        else:
            line = get_line(points[i], points[i+1])
            lines.append(line)
    return lines


def divide_contours_by_lines(contours, lines, threshold=15):
    ''' Divide contours based on the lines '''
    # A dictionary with keys line_1, line_2, ... and values as a list of contours initially empty
    contours_by_lines = {f'line_{i+1}': [] for i in range(len(lines))}
    for contour in contours:
        # calculate the centroid of the contour
        centroid = get_centroid(contour)
        for i, line in enumerate(lines):
            # if the distance between the centroid and the line is less than a threshold, add the contour to the current line
            if distance_point_to_line(centroid, line) < threshold:
                contours_by_lines[f'line_{i+1}'].append(contour)
                break
    return contours_by_lines


def extract_regions_of_interest(image, contours):
    ''' Extract regions of interest from a given image '''
    regions = []
    for contour in contours:
        # get the bounding box of the contour
        x, y, w, h = cv2.boundingRect(contour)
        # extract the region of interest from the image
        region = image[y:y+h, x:x+w]
        regions.append(region)
    return regions


def draw_line(image, line):
    ''' Draw a line in a given image '''
    slope, intercept = line
    x1 = 0
    y1 = int(slope * x1 + intercept)
    x2 = image.shape[1]
    y2 = int(slope * x2 + intercept)
    cv2.line(image, (x1, y1), (x2, y2), (35, 255, 55), 1)


def draw_on_image(image, contours, centroid=None, lines=None):
    ''' Draw contours, centroids and lines in a given image '''
    image_contours = image.copy()
    cv2.drawContours(image_contours, contours, -1, (0, 255, 0), 2)
    # draw centroids
    if centroid:
        for cnt in contours:
            x, y = get_centroid(cnt)
            cv2.circle(image_contours, (x, y), 2, (255, 0, 0), -1)
    for line in lines:
        draw_line(image_contours, line)
    return image_contours


if __name__ == '__main__':
    images = load_images('./Práctica2_Datos_Alumnos/test_ocr_panels/')
    for image in images:
        binary = convert_to_binary(image)
        contours = get_contours(binary)
        lines = get_lines(contours)
        contours_by_lines = divide_contours_by_lines(contours, lines)
        for line, line_contours in contours_by_lines.items():
            regions = extract_regions_of_interest(image, line_contours)
            # TODO: classify the regions
