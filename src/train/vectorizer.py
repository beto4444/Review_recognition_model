import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import os
from pathlib import Path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = Path(ROOT_DIR).parents[1]
VECTORIZER_PATH = os.path.join(ROOT_DIR, 'models/vectorizer.pkl')
def save_vectorizer(vectorizer, path):
    with open(path, 'wb') as f:
        pickle.dump(vectorizer, f)

def to_tf_idf(df: pd.DataFrame):
    vectorizer = TfidfVectorizer(max_features=5000, min_df = 5, max_df = 0.7)
    X = vectorizer.fit_transform(df['review']).toarray()
    save_vectorizer(vectorizer, VECTORIZER_PATH)
    return X

def to_bag_of_words(df: pd.DataFrame):
    vectorizer = CountVectorizer(max_features=5000, min_df = 5, max_df = 0.7)
    X = vectorizer.fit_transform(df['review']).toarray()
    save_vectorizer(vectorizer, VECTORIZER_PATH)

def saved_vect_transform(df: pd.DataFrame):
    with open(VECTORIZER_PATH, 'rb') as f:
        vectorizer = pickle.load(f)
    X = vectorizer.transform(df['review']).toarray()
    return X