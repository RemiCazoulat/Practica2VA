# @brief NormalBayesClassifierWithLBP
# This class implements a classifier for optical character recognition using Local Binary Patterns (LBP)
# as feature descriptors combined with a Normal Bayes Classifier from OpenCV.

import cv2
import numpy as np
from skimage import feature
import string

class NormalBayesClassifierWithLBP:
    """
    A classifier for Optical Character Recognition that uses Local Binary Patterns (LBP) 
    for feature extraction and a Normal Bayes Classifier for making predictions.
    """

    def __init__(self):
        # Initializes the classifier; here, we use the Normal Bayes Classifier from OpenCV's machine learning module.
        self.classifier = cv2.ml.NormalBayesClassifier_create()

    def extract_lbp_features(self, image):
        # This method extracts Local Binary Patterns (LBP) features from a single image.
        # LBP is a type of visual descriptor used for classification in computer vision.
        # Parameters: P (number of circularly symmetric neighbour set points), R (radius of circle).
        image = cv2.resize(image, (64, 64))
        lbp_image = feature.local_binary_pattern(image, P=22, R=4, method='uniform')
        lbp_hist, _ = np.histogram(lbp_image.ravel(), bins=np.arange(0, 27), range=(0, 26))
        lbp_hist = lbp_hist.astype("float")
        lbp_hist /= (lbp_hist.sum() + 1e-6)  # Normalize the histogram
        return lbp_hist  # Return the normalized histogram as the feature set.

    def train(self, X, labels):
        """
        Trains the OCR classifier using a list of images and their associated labels.
        This method first extracts LBP features from each image and then trains the classifier.

        :param X: A list of images, each as a numpy array.
        :param labels: The corresponding labels for each image in X.
        """
        X_features = [self.extract_lbp_features(image) for image in X]
        X_features = np.array(X_features, dtype=np.float32)
        labels = np.array(labels, dtype=np.int32)

        print("[DEBUG] Starting classifier training...")
        self.classifier.train(X_features, cv2.ml.ROW_SAMPLE, labels)
        print("[DEBUG] Classifier training completed.")

    def predict(self, images):
        """
        Predicts the class labels for a given array of images using the trained classifier.
        This method first extracts LBP features from each image and then uses the classifier to predict the labels.

        :param images: Array of images for which to predict class labels.
        :return: List of predicted class labels as integers.
        """
        predicted_labels = []
        for image in images:
            lbp_features = self.extract_lbp_features(image)
            lbp_features = np.array([lbp_features], dtype=np.float32)  # Prepare feature array for prediction
            _, predicted_y = self.classifier.predict(lbp_features)
            predicted_labels.append(int(predicted_y[0, 0]))  # Append the predicted class label as an integer
        return predicted_labels
