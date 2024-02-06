import pandas as pd
import re
import requests
import os
import zipfile
import json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

nltk.download('stopwords')


CONFIG = []

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def download_data(url_train, url_test):
    if not os.path.exists('data/raw'):
        os.makedirs('data/raw')

    train_data = requests.get(url_train)

    with open('data/raw/train.zip', 'wb') as f:
        f.write(train_data.content)

    test_data = requests.get(url_test)

    with open('data/raw/test.zip', 'wb') as f:
        f.write(test_data.content)

    print('Data downloaded')

    with zipfile.ZipFile('data/raw/train.zip', 'r') as zip_ref:
        zip_ref.extractall('data/raw')

    with zipfile.ZipFile('data/raw/test.zip', 'r') as zip_ref:
        zip_ref.extractall('data/raw')
                        

    print('Data unzipped')

    os.rename('data/raw/final_project_test_dataset/test.csv', 'data/raw/test.csv')
    os.rename('data/raw/final_project_train_dataset/train.csv', 'data/raw/train.csv')

    os.remove('data/raw/train.zip')
    os.remove('data/raw/test.zip')
    os.rmdir('data/raw/final_project_test_dataset')
    os.rmdir('data/raw/final_project_train_dataset')
    print('Zip files removed')

    print('Data ready')

class TextProcessing:
    def clean_url(text: str):
        return re.sub(r'http\S+', '', text)

    def clean_html(text: str):
        return re.sub(r'<.*?>', '', text)
    
    def clean_non_alphabetic(text: str):
        return re.sub(r'[^a-zA-Z]', ' ', text)
    
    def clean_multiple_spaces(text: str):
        return re.sub(r'\s+', ' ', text)

    def to_lower(text: str):
        return text.lower()
    
    def tokenize(text: str)-> list:
        return text.split()
    
    def remove_stopwords(tokens: list)-> list:
        stop_words = set(stopwords.words('english'))
        return [word for word in tokens if word not in stop_words]
    
    def lemmatize(tokens: list)-> list:
        lemmatizer = WordNetLemmatizer()
        return [lemmatizer.lemmatize(word, pos = 'v') for word in tokens]
    
    def stem(tokens: list)-> list:
        stemmer = PorterStemmer()
        return [stemmer.stem(word) for word in tokens]
    
    def clean_short_words(tokens: list, min_length: int)-> list:
        return [word for word in tokens if len(word) >= min_length]

    def join_tokens(tokens: list)-> str:
        return ' '.join(tokens)

    def remove_words(text: str, rm_words: list)-> str:
        return ' '.join([t for t in text.split() if t not in rm_words])


    def process_text(text: str, mode: int)-> str: #mode 0: lemmatize, mode 1: stem
        text = TextProcessing.clean_url(text)
        text = TextProcessing.clean_html(text)
        text = TextProcessing.clean_non_alphabetic(text)
        text = TextProcessing.clean_multiple_spaces(text)
        text = TextProcessing.to_lower(text)
        tokens = TextProcessing.tokenize(text)
        tokens = TextProcessing.remove_stopwords(tokens)
        if mode == 0:
            tokens = TextProcessing.lemmatize(tokens)
        else:
            tokens = TextProcessing.stem(tokens)
        tokens = TextProcessing.clean_short_words(tokens, 3)
        text = TextProcessing.join_tokens(tokens)
        return TextProcessing.remove_words(text, ['movie', 'film', 'one', 'even', 'get'])


if __name__ == '__main__':
    CONFIG = load_config('config.json')
    download_data(CONFIG['train_data_url'], CONFIG['test_data_url'])

    train = pd.read_csv('data/raw/train.csv')
    test = pd.read_csv('data/raw/test.csv')

    if not os.path.exists('data/processed'):
        os.makedirs('data/processed')

    print('text processing started..')
    train_lemmatized = train['review'].apply(TextProcessing.process_text, mode=0)

    pd.concat([train_lemmatized, train['sentiment'].map({'positive': 1, 'negative': 0})], axis=1).to_csv(
        'data/processed/train_lemmatized.csv', index=False)

    test_lemmatized = test['review'].apply(TextProcessing.process_text, mode=0)
    pd.concat([test_lemmatized, test['sentiment'].map({'positive': 1, 'negative': 0})], axis=1).to_csv(
        'data/processed/inference_lemmatized.csv', index=False)

    print('Lemmatized files saved')
    train_stemed = train['review'].apply(TextProcessing.process_text, mode=1)
    pd.concat([train_stemed, train['sentiment'].map({'positive': 1, 'negative': 0})], axis=1).to_csv(
        'data/processed/train_stemed.csv', index=False)

    test_stemed = test['review'].apply(TextProcessing.process_text, mode=1)
    pd.concat([test_stemed, test['sentiment'].map({'positive': 1, 'negative': 0})], axis=1).to_csv(
        'data/processed/inference_stemed.csv', index=False)

    print('Stemed files saved')