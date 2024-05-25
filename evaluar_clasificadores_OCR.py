# Asignatura de Visión Artificial (URJC). Script de evaluación.


import argparse

#import panel_det
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import time
import seaborn as sns

from lda_normal_bayes_classifier import LdaNormalBayesClassifier
from lbp_normal_bayes_classifier import NormalBayesClassifierWithLBP
from hog_normal_bayes_classifier import HOGNormalBayesClassifier

def getImagesAndLabels(path) :
    """.
    Given a path, return all the images and their corresponding labels. 
    """
    images = []
    labels = []
    #path = "train_ocr"
    label = 0
    
    #print("[LOADING] loading training data ...")
    for nbr in range(10):
        currentPath = path + "/" + str(nbr) #+ "/" #+ "*.png"
        print("loading number " + str(nbr))
        files = os.listdir(currentPath)
        for file in files:
            imgPath = currentPath + "/" + file
            cv_img = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
            images.append(cv_img)
            labels.append(label)
        label = label + 1

    types = ["maj", "min"]
    for type in types :
        pathType = path + "/" + type
        letters = os.listdir(pathType)
        for letter in letters:
            print("loading letter " + str(letter))
            letterPath = pathType + "/" + letter
            files = os.listdir(letterPath)
            for file in files:
                imgPath = letterPath + "/" + file
                cv_img = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
                images.append(cv_img)
                labels.append(label)
            label = label + 1
 
    print("number of images : " + str(len(images)) + ", number we want : " + str(625 * 10 + 26 * 2 * 625))
    return images, labels


def preparingData(images)  :
    lines = []
    i = 1
    for image in images:
        imgNorm = cv2.adaptiveThreshold(image, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,3 ,0)
        """
        bounds, _ = cv2.findContours(imgNorm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        x = 0
        y = 0
        w = 0
        h = 0
        for bound in bounds:
            x1, y1, w1, h1 = cv2.boundingRect(bound)
            if w1 * h1 > w * h :
                x = x1
                y = y1
                w = w1
                h = h1
        imgNorm = imgNorm[y:y+h, x:x+w]
        """
        imgNorm = cv2.resize(imgNorm, (25, 25))
        line = np.array(imgNorm.flatten())
        lines.append(line)
        print("image " + str(i) + "/" + str(len(images)) + " done.")
        i += 1
    return lines

def preparingData2(images):
    processed_images = []
    flat_images = []  # This will store the flattened images for LDA
    for image in images:
        # Apply adaptive thresholding
        imgNorm = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

        # Resize the image to a consistent size
        imgNorm = cv2.resize(imgNorm, (25, 25))

        # Append the 2D image for LBP extraction
        processed_images.append(imgNorm)

        # Flatten and append for LDA
        flat_images.append(imgNorm.flatten())

    return processed_images, flat_images


def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.colormaps['Blues']):
    '''
    Given a confusión matrix in cm (np.array) it plots it in a fancy way.
    '''
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(cm.shape[0])
    plt.xticks(tick_marks, range(cm.shape[0]))
    plt.yticks(tick_marks, range(cm.shape[0]))
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    ax = plt.gca()
    width = cm.shape[1]
    height = cm.shape[0]

    for x in range(width):
        for y in range(height):
            ax.annotate(str(cm[y,x]), xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center')


def evaluate_classifier(classifier, train_X, train_y, test_X, test_y, classifier_name):
    start_time = time.time()
    classifier.train(train_X, train_y)
    training_time = time.time() - start_time
    
    start_time = time.time()
    predicted_y = classifier.predict(test_X)
    prediction_time = time.time() - start_time

    # Calculate metrics
    accuracy = accuracy_score(test_y, predicted_y)
    conf_matrix = confusion_matrix(test_y, predicted_y)
    report = classification_report(test_y, predicted_y, output_dict=True)  # Change to output_dict for easier plotting

    metrics = {
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix,
        'classification_report': report,
        'training_time': training_time,
        'prediction_time': prediction_time
    }

    save_evaluation_results("the_results.txt", classifier_name, metrics, training_time, prediction_time)
    return metrics

def save_evaluation_results(file_path, classifier_name, metrics, training_time, prediction_time):
    with open(file_path, 'a') as file:
        file.write(f"Results for {classifier_name}\n")
        file.write(f"Training time: {training_time:.2f} seconds\n")
        file.write(f"Prediction time: {prediction_time:.2f} seconds\n")
        file.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        file.write("Confusion Matrix:\n")
        confusion_matrix = metrics['confusion_matrix']
        for row in confusion_matrix:
            file.write(' '.join(f"{num:4d}" for num in row) + "\n")
        
        file.write("\nClassification Report:\n")
        # Convert classification report dictionary into a readable string
        for label, values in metrics['classification_report'].items():
            if isinstance(values, dict):  # Ensuring it's not the 'accuracy' key, which is a single float
                report_line = f"{label}:\n"
                for metric, value in values.items():
                    report_line += f"    {metric}: {value:.2f}\n"
                file.write(report_line)
            else:
                file.write(f"{label}: {values:.2f}\n")
        
        file.write("="*80 + "\n\n")

def plot_accuracy(results):
    """ Plot bar chart of accuracy for each classifier. """
    accuracies = [results[key]['accuracy'] for key in results]
    classifiers = list(results.keys())
    
    plt.figure(figsize=(8, 5))
    plt.bar(classifiers, accuracies, color='skyblue')
    plt.xlabel('Classifier')
    plt.ylabel('Accuracy')
    plt.title('Classifier Accuracy Comparison')
    plt.ylim(0, 1)
    plt.show()

def plot_confusion_matrices(results):
    """ Plot confusion matrices for each classifier. """
    fig, axes = plt.subplots(nrows=1, ncols=len(results), figsize=(15, 5))
    for idx, (key, value) in enumerate(results.items()):
        sns.heatmap(value['confusion_matrix'], annot=True, fmt='d', ax=axes[idx], cmap='Blues')
        axes[idx].set_title(f'{key} Confusion Matrix')
        axes[idx].set_xlabel('Predicted Labels')
        axes[idx].set_ylabel('True Labels')
    plt.tight_layout()
    plt.show()

def plot_performance_metrics(results):
    classifiers = list(results.keys())
    metrics = ['precision', 'recall', 'f1-score']
    # Prepare data for each metric
    data = {metric: [results[cls]['classification_report']['weighted avg'][metric] for cls in classifiers] for metric in metrics}

    # Plot each metric in a separate subplot
    fig, axes = plt.subplots(1, len(metrics), figsize=(18, 5), sharey=True)
    for ax, metric in zip(axes, metrics):
        sns.barplot(x=classifiers, y=data[metric], ax=ax)
        ax.set_title(f'{metric.capitalize()} Comparison')
        ax.set_ylim(0, 1)  # Adjust y-axis for percentage
        ax.set_ylabel(metric.capitalize())
        ax.set_xlabel('Classifier')

    plt.tight_layout()
    plt.show()

# Call plot functions in main after collecting results

if __name__ == "__main__":
    # 1) Cargar las imágenes de entrenamiento y sus etiquetas. 
    # También habrá que extraer los vectores de características asociados (en la parte básica 
    # umbralizar imágenes, pasar findContours y luego redimensionar)
    print("[LOADING] getting training & validation images ...")
    
    trainPath = "train_ocr"
    validationPath = "validation_ocr"
    train_X, train_y = getImagesAndLabels(trainPath)
    val_X, val_y = getImagesAndLabels(validationPath)

    # 2) Load training and validation data
    print("[LOADING] loading training & validation data ...")

    #train_X = preparingData(train_X)
    #val_X = preparingData(val_X)

    train_processed_X, train_flat_X = preparingData2(train_X)  # Process images for both use cases
    val_processed_X, val_flat_X = preparingData2(val_X)  # We are using the second output for LDA-based models

    #val_processed_X, val_flat_X = preparingData2(val_X)
    # También habrá que extraer los vectores de características asociados (en la parte básica 
    # umbralizar imágenes, pasar findContours y luego redimensionar)
    
    lda_classifier = LdaNormalBayesClassifier()
    lbp_classifier = NormalBayesClassifierWithLBP()
    hog_classifier = HOGNormalBayesClassifier()
    results = {}

    #deleting the what the file contained
    open('the_results.txt', 'w').close()
    results['LDA'] = evaluate_classifier(lda_classifier, train_flat_X, train_y, val_flat_X, val_y, "LDA Normal Bayes Classifier")
    results['LBP'] = evaluate_classifier(lbp_classifier, train_processed_X, train_y, val_processed_X, val_y, "LBP Normal Bayes Classifier")
    results['HOG'] = evaluate_classifier(hog_classifier, train_processed_X, train_y, val_processed_X, val_y, "HOG Normal Bayes Classifier")

    plot_accuracy(results)
    plot_confusion_matrices(results)
    plot_performance_metrics(results)

    """
    # 3) Entrenar clasificador

    LDA = LdaNormalBayesClassifier()
    LDA.train(train_flat_X, train_y)

    # 4) Ejecutar el clasificador sobre los datos de validación    
    predicted_labels = LDA.predict(val_flat_X)
    print("[DEBUG] testing done.")

   
   
    # 5) Evaluar los resultados
    #accuracy = sklearn.metrics.accuracy_score(gt_labels, predicted_labels)
    #print("Accuracy = ", accuracy)
    succes_val = np.sum(1 * (val_y == predicted_labels)) / len(val_y)
    print("Test accuracy of LDA: " + str(succes_val * 100) + "%")
    
    
    # 3) Entrenar clasificador

    lbp_classifier = NormalBayesClassifierWithLBP()
    lbp_classifier.train(np.array(train_processed_X), np.array(train_y))

    # 4) Ejecutar el clasificador sobre los datos de validación    
    # Execute the classifier over validation data and collect predictions
    predicted_labels = []
    for image in val_processed_X:
        predicted_label = lbp_classifier.predict(image)
        predicted_labels.append(predicted_label)
    print("[DEBUG] testing done.")

    # 5) Evaluar los resultados
    # Evaluate the results
    predicted_labels = np.array(predicted_labels)
    accuracy = np.mean(predicted_labels == val_y)
    print(f"Test accuracy of LBP: {accuracy * 100:.2f}%")

    
    # 3) Entrenar clasificador

    hog_classifier = HOGNormalBayesClassifier()
    hog_classifier.train(np.array(train_processed_X), np.array(train_y))

    # 4) Ejecutar el clasificador sobre los datos de validación    
    # Execute the classifier over validation data and collect predictions
    predicted_labels = []
    for image in val_processed_X:
        predicted_label = hog_classifier.predict(image)
        predicted_labels.append(predicted_label)
    print("[DEBUG] testing done.")

    # 5) Evaluar los resultados
    # Evaluate the results
    predicted_labels = np.array(predicted_labels)
    accuracy = np.mean(predicted_labels == val_y)
    print(f"Test accuracy of HOG: {accuracy * 100:.2f}%")
"""
    



