# @brief HOGNormalBayesClassifier


import cv2
import numpy as np
from skimage.feature import hog



class HOGNormalBayesClassifier: 
    """
        A classifier for Optical Character Recognition that uses Histogram of Oriented Gradients (HOG) 
        for feature extraction and a Normal Bayes Classifier for making predictions.
    """ 
    def __init__(self):
        self.classifier = cv2.ml.NormalBayesClassifier_create()

    @staticmethod
    def extract_hog_features(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), orientations=20):
        # This method extracts Histogram of Oriented Gradients (HOG) features from a single image.
        # HOG is a type of feature descriptor used in computer vision and image processing for the purpose of object detection.
        # Parameters: 
        #   pixels_per_cell: Size (in pixels) of a cell.
        #   cells_per_block: Number of cells in each block.
        #   orientations: Number of orientation bins.
        if image.ndim > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (64, 64))
        features = hog(image, orientations=orientations, pixels_per_cell=(16, 16),
            cells_per_block=(1, 1), feature_vector=True)
        return features

    def train(self, X, y):
        hog_features = [self.extract_hog_features(img) for img in X]
        print("done with the features")
        hog_features = np.array(hog_features, dtype=np.float32)
        print("features are in an array, ready to train")
        self.classifier.train(hog_features, cv2.ml.ROW_SAMPLE, np.array(y, dtype=np.int32))
        print("training is done")

    def predict(self, images):
        """
        Predicts the class labels for a given array of images using the trained classifier.
        This method first extracts HOG features from each image and then uses the classifier to predict the labels.

        :param images: Array of images for which to predict class labels.
        :return: List of predicted class labels as integers.
        """
        predicted_labels = []
        for image in images:
            hog_features = self.extract_hog_features(image)
            hog_features = np.array([hog_features], dtype=np.float32)  # Prepare feature array for prediction
            _, y_pred = self.classifier.predict(hog_features)
            predicted_labels.append(int(y_pred[0, 0]))  # Append the predicted class label as an integer
        return predicted_labels