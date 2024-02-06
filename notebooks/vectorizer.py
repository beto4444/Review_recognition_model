import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

def to_tf_idf(df: pd.DataFrame)-> np.ndarray:
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['review'])
    return X

def to_bag_of_words(df: pd.DataFrame)-> np.ndarray:
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['review'])
    return X