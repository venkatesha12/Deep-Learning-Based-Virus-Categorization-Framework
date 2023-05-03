
import sys
import pandas as pd
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
import pickle

class Prediction:

    def predict(calls="GetSystemDirectoryA IsDBCSLeadByte LocalAlloc CreateSemaphoreW CreateSemaphoreA"):
        modelfile="rf_model.sav"

        #train_news = pd.read_csv(train_file)
        #test_ = pd.read_csv(test_file)
        #tfidf = TfidfVectorizer(stop_words='english',use_idf=True,smooth_idf=True) #TF-IDF
        #knn_pipeline = Pipeline([('lrgTF_IDF', tfidf), ('lrg_mn', KNeighborsClassifier())])

        #filename = 'nb_model.sav'
        filename=modelfile
        #pickle.dump(knn_pipeline.fit(train_news['review'], train_news['sentiment']), open(filename, 'wb'))
        train = pickle.load(open(filename, 'rb'))
        predicted_class = train.predict([calls])
        print(predicted_class[0])
        print("Successfully Predicted")
        return predicted_class[0]
        
if __name__ == "__main__":
    Prediction.predict()
    
    

