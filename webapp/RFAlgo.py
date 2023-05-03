import sys
import pandas as pd
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

class RFAlgo:

    def classification(train_file='malware_API_dataset.csv'):

        train_news = pd.read_csv(train_file)
        tfidf = TfidfVectorizer(stop_words='english',use_idf=True,smooth_idf=True) #TF-IDF
        print("RF Classifier Classification")
        nb_pipeline = Pipeline([('lrgTF_IDF', tfidf), ('lrg_mn', RandomForestClassifier())])#0.7045
 
        filename = 'rf_model.sav'
        pickle.dump(nb_pipeline.fit(train_news['API_Calls'], train_news['Malware']), open(filename, 'wb'))

        print("RF Classifier Successfully Trained")


if __name__ == "__main__":
    pass

