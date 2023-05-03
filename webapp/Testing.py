
import sys
import pandas as pd
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
import pickle

class Testing:

    def predict(modelfile,test_file='Testing.csv'):

        test_ = pd.read_csv(test_file)
        filename=modelfile
        train = pickle.load(open(filename, 'rb'))
        predicted_class = train.predict(test_["API_Calls"])
        print("Successfully Predicted")
        
        test_data = pd.read_csv(test_file)
        res = Testing.model_assessment(test_data['Malware'], predicted_class)
        return res
    

    def model_assessment(y_test, predicted_class):
        print('accuracy')
        # Accuracy = (TP + TN) / ALL
        accuracy = accuracy_score(y_test, predicted_class)
        print(accuracy)
        return accuracy


if __name__ == "__main__":
    l=Testing.predict('nn_model.sav')
    print(l)
    

