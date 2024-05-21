# Asignatura de Visión Artificial (URJC). Script de evaluación.


import argparse

#import panel_det
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import sklearn

from lda_normal_bayes_classifier import LdaNormalBayesClassifier

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

    train_X = preparingData(train_X)
    val_X = preparingData(val_X)

    # También habrá que extraer los vectores de características asociados (en la parte básica 
    # umbralizar imágenes, pasar findContours y luego redimensionar)

    # 3) Entrenar clasificador

    LDA = LdaNormalBayesClassifier()
    LDA.train(train_X, train_y)

    # 4) Ejecutar el clasificador sobre los datos de validación    
    predicted_labels = LDA.predict(val_X)
    print("[DEBUG] testing done.")

   

   
    # 5) Evaluar los resultados
    #accuracy = sklearn.metrics.accuracy_score(gt_labels, predicted_labels)
    #print("Accuracy = ", accuracy)
    succes_val = np.sum(1 * (val_y == predicted_labels[:,0])) / len(val_y)
    print("Test accuracy: " + str(succes_val * 100) + "%")


