import os
import cv2
from matplotlib import pyplot as plt
from lda_normal_bayes_classifier import LdaNormalBayesClassifier
from evaluar_clasificadores_OCR import getImagesAndLabels, preparingData


def load_image(path):
    ''' Load an image from a given path'''
    if not os.path.exists(path):
        raise FileNotFoundError('The given path does not exist')
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    # convert from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def convert_to_binary(image, thresoldType=cv2.ADAPTIVE_THRESH_MEAN_C, blockSize=9, C=5):
    ''' Convert an image to binary '''
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    binary = cv2.adaptiveThreshold(
        gray, 255, thresoldType, cv2.THRESH_BINARY_INV, blockSize, C)
    return binary


def get_contours(binary, min_contour_area=50, max_contour_area=700, min_aspect_ratio=0.25, max_aspect_ratio=1.1):
    ''' Get contours in a given binary image '''
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
    if x2 - x1 == 0:
        return (0, 0)
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
    points = [get_centroid(cnt) for cnt in contours]
    # remove duplicates
    points = list(set(points))
    # if there are less than two centroids, return an empty list
    if len(points) < 2:
        return []
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
    # a dictionary with keys line_1, line_2, ... and values as a list of contours initially empty
    contours_by_lines = {f'line_{i+1}': [] for i in range(len(lines))}
    for contour in contours:
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
        x, y, w, h = cv2.boundingRect(contour)
        region = image[y:y+h, x:x+w]
        # convert the region to the format that the classifier expects
        region = convert_to_binary(region)
        region = cv2.resize(region, (25, 25))
        region = region.flatten().reshape(1, -1)
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
    if centroid:
        for cnt in contours:
            x, y = get_centroid(cnt)
            cv2.circle(image_contours, (x, y), 2, (255, 0, 0), -1)
    for line in lines:
        draw_line(image_contours, line)
    return image_contours


def create_results(images_path, results_path, classifier):
    ''' Create a results file '''
    with open(results_path, 'w') as f:
        for image_name in os.listdir(images_path):
            if not image_name.endswith('.png'):
                continue
            image_str = create_string(images_path, image_name, classifier)
            f.write(image_str)


def create_string(images_path, image_name, classifier):
    ''' Create the prediction string for a given image '''
    labels = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    image = load_image(images_path + image_name)
    x = image.shape[0] - 1
    y = image.shape[1] - 1
    image_str = f'{image_name};0;0;{x};{y};1;1;'
    binary = convert_to_binary(image)
    contours = get_contours(binary)
    lines = get_lines(contours)
    contours_by_lines = divide_contours_by_lines(contours, lines)
    for line_contours in contours_by_lines.values():
        if image_str[-1] != ';':
            image_str += '+'
        regions = extract_regions_of_interest(image, line_contours)
        for region in regions:
            prediction = classifier.predict(region)
            image_str += labels[prediction[0]]
    return image_str+'\n'


def show_results(image_path):
    ''' Show the results visually on a given image '''
    image = load_image(image_path)
    binary = convert_to_binary(image)
    contours = get_contours(binary)
    centroids = [get_centroid(cnt) for cnt in contours]
    lines = get_lines(contours)
    image_contours = draw_on_image(image, contours, centroids, lines)
    plt.imshow(image_contours)
    plt.show()


if __name__ == '__main__':
    images_path = './test_ocr_panels/'
    training_path = './train_ocr/'
    # Load the training data
    train_X, train_y = getImagesAndLabels(training_path)
    train_X = preparingData(train_X)
    # Train the classifier
    classifier = LdaNormalBayesClassifier()
    classifier.train(train_X, train_y)
    # Create the results file
    create_results(
        images_path, './resultado.txt', classifier)
    # Show the results on a given image
    # show_results(images_path + '00041_0.png')
