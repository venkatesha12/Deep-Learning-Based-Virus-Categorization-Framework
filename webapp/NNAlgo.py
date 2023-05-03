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

class NNAlgo:

    def classification(train_file='malware_API_dataset.csv'):

        train_news = pd.read_csv(train_file)
        tfidf = TfidfVectorizer(stop_words='english',use_idf=True,smooth_idf=True) #TF-IDF
        print("DT Classifier Classification")
        #nb_pipeline = Pipeline([('lrgTF_IDF', tfidf), ('lrg_mn', BernoulliNB())])
        #nb_pipeline = Pipeline([('lrgTF_IDF', tfidf), ('lrg_mn', MultinomialNB())])#0.3225
        #nb_pipeline = Pipeline([('lrgTF_IDF', tfidf), ('lrg_mn', KNeighborsClassifier())])#0.438
        #nb_pipeline = Pipeline([('lrgTF_IDF', tfidf), ('lrg_mn', RandomForestClassifier())])#0.7045
        #nb_pipeline = Pipeline([('lrgTF_IDF', tfidf), ('lrg_mn', AdaBoostClassifier())])#0.102
        nb_pipeline = Pipeline([('lrgTF_IDF', tfidf), ('lrg_mn', MLPClassifier())])#0.707

        filename = 'nn_model.sav'
        pickle.dump(nb_pipeline.fit(train_news['API_Calls'], train_news['Malware']), open(filename, 'wb'))

        print(" NN Classifier Successfully Trained")


if __name__ == "__main__":
    pass

