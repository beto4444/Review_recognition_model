import pandas as pd
import os
from pathlib import Path
import numpy as np
import json
from vectorizer import to_tf_idf
from vectorizer import to_bag_of_words
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

import tensorflow as tf
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = Path(ROOT_DIR).parents[1]
config_path = os.path.join(ROOT_DIR, 'config.json')

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

train = []
processed = []
if __name__ == '__main__':
    
    CONFIG = load_config(config_path)
    if CONFIG['vectorizer'] == 'TF-IDF' and CONFIG['reduction'] == 'Lemmatization':
        train = pd.read_csv(os.path.join(ROOT_DIR, 'data/processed/train_lemmatized.csv'))
        processed = to_tf_idf(train)
    elif CONFIG['vectorizer'] == 'TF-IDF' and CONFIG['reduction'] == 'stemming':
        train = pd.read_csv(os.path.join(ROOT_DIR, 'data/processed/train_stemed.csv'))
        processed = to_tf_idf(train)
    elif CONFIG['vectorizer'] == 'bag_of_words' and CONFIG['reduction'] == 'Lemmatization':
        train = pd.read_csv(os.path.join(ROOT_DIR, 'data/processed/train_lemmatized.csv'))
        processed = to_bag_of_words(train)
    elif CONFIG['vectorizer'] == 'bag_of_words' and CONFIG['reduction'] == 'stemming':
        train = pd.read_csv(os.path.join(ROOT_DIR, 'data/processed/train_stemed.csv'))
        processed = to_bag_of_words(train)
    else:
        print('Invalid configuration')
    X_train, X_test, y_train, y_test = train_test_split(processed, train['sentiment'], test_size=0.2, random_state=0)
    
    if not os.path.exists(os.path.join(ROOT_DIR, 'models')):
        os.makedirs(os.path.join(ROOT_DIR, 'models'))
    logistic_regression_path = os.path.join(ROOT_DIR, 'models/model1.pkl')
    rn_forset_path = os.path.join(ROOT_DIR, 'models/model2.pkl')

    print('Training model 1 - Random Forest')
    random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
    random_forest.fit(X_train, y_train)
    y_pred = random_forest.predict(X_test)


    print('Random Forest Accuracy:', accuracy_score(y_test, y_pred))
    log_reg = LogisticRegression()
    print('Training model - Logistic Regression')
    log_reg.fit(X_train, y_train)
    print('Logistic Regression Accuracy:', log_reg.score(X_test, y_test))

    with open(logistic_regression_path, 'wb') as f:
        pickle.dump(log_reg, f)
    with open(rn_forset_path, 'wb') as f:
        pickle.dump(random_forest, f)

        









