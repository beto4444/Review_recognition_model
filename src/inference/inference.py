from sklearn.linear_model import LogisticRegression 
import pickle

import os
import json
from vectorizer import saved_vect_transform
from pathlib import Path
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)

import pandas as pd
def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def import_model(file_path: str)-> LogisticRegression:
    clf = pickle.load(open(file_path, 'rb'))
    return clf

def predict_review(model: LogisticRegression, review: str)-> int:
    return model.predict(review)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = Path(ROOT_DIR).parents[1]
if __name__ == '__main__':
    model_path = os.path.join(ROOT_DIR, 'models/model1.pkl')
    model = import_model(model_path)
    
    config_path = os.path.join(ROOT_DIR, 'config.json')
    CONFIG = load_config(config_path)
    if CONFIG['reduction'] == 'Lemmatization':
        data_path = os.path.join(ROOT_DIR, 'data/processed/inference_lemmatized.csv')
    elif CONFIG['reduction'] == 'stemming':
        data_path = os.path.join(ROOT_DIR, 'data/processed/inference_stemed.csv')

    data = pd.read_csv(data_path)

    processed = saved_vect_transform(data)

    print('Inference started..')

    y_pred = model.predict(processed)
    print('Inference completed..')

    print('Saving predictions..')
    predictions = pd.Series(y_pred, name='sentiment')
    predictions.to_csv(os.path.join(ROOT_DIR, 'app/results/predictions.csv'), index=False)

    print('Predictions saved to /app/results/predictions.csv')

    print(accuracy_score(y_pred, data['sentiment']))
    print(confusion_matrix(y_pred, data['sentiment']))
    print(classification_report(y_pred, data['sentiment']))
          

    report = classification_report(y_pred, data['sentiment'])

    results_path = "/app/results/summary.txt"
    with open(results_path, "w") as file:
        file.write("Summary of classification:\n")
        file.write("--------------------------------\n")
        file.write(report)

    print(f"Report saved in {results_path}")
