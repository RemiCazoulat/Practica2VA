# @brief LdaNormalBayesClassifier


# A continuación se presenta un esquema de la clase necesaria para implementar el clasificador
# propuesto en el Ejercicio1 de la práctica. Habrá que terminarla

import cv2
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
import string
class LdaNormalBayesClassifier():
    """
    Classifier for Optical Character Recognition using LDA and the Bayes with Gaussian classfier.
    """


    #args ? : ocr_char_size
    def __init__(self):
        #super().__init__(ocr_char_size)
        self.lda = LinearDiscriminantAnalysis()

        self.classifier = cv2.ml.NormalBayesClassifier_create()


    def train(self, X, labels):
        """.
        Given character images in a dictionary of list of char images of fixed size flattened in one vector, 
        train the OCR classifier. The dictionary keys are the class of the list of images flattened.
        (or corresponding char).

        :images_dict is a dictionary of images (name of the images is the key)
        """       
        # Perform LDA training
        X = np.array(X,dtype=np.float32)
        self.lda.fit(X,labels)
        Xreduced = self.lda.transform(X)

        print("[DEBUG] reducing done.")

        # Perform Classifier training
        #Xreduced32 = Xreduced.astype(np.float32)

        #X_train,X_test,y_train,y_test=train_test_split(Xreduced,labels,test_size=0.2,stratify = labels)
        print("[DEBUG] splitting done.")
        print("[DEBUG] creating done.")
        self.classifier.train(np.float32(Xreduced), cv2.ml.ROW_SAMPLE, np.int32(labels))
        print("[DEBUG] training done.")

    def predict(self, images):
        """.
       
        
        """
        img = self.lda.transform(images)
        _, predicted_y = self.classifier.predict(np.float32(img))
        #acierto_test = np.sum(1 * (np.array(y_test) == predicted_test[:, 0])) / len(y_test)
        #print("Test accuracy: {:.2f}%".format(acierto_test * 100))
        
        #y = ... # Obtain the estimated label by the LDA + Bayes classifier
        predicted_y = predicted_y[:,0]
        return predicted_y



