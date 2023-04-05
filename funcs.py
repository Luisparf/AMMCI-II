# -*- coding: utf-8 -*-
import re
import string
import nltk
from nltk.corpus import stopwords
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# Pré processamento #################################################################################

def clean_text(text):
    # Remove pontuação
    if isinstance(text, str) and not pd.isnull(text): # se o texto é número ou nulo, ignora

        # print(f'{text}\n\n')
        text = text.translate(str.maketrans("", "", string.punctuation))
        # Converte para minusculos
        text = text.lower()
        # Remove números
        text = re.sub(r'\d+', '', text)
        stop_words = set(stopwords.words('english'))
        text_tokens = nltk.word_tokenize(text)
        filtered_text = [word for word in text_tokens if word not in stop_words]
        text = ' '.join(filtered_text)
    else:
        text = ''
    return text

######################################################################################################

def load_data():
    
    data = pd.read_csv('train.csv')
    train_data = data['selected_text'].apply(lambda x: clean_text(x))
    text_data = data['text'].apply(lambda x: clean_text(x))
    train_target = data['sentiment']
    return train_data, text_data, train_target

######################################################################################################

def run(X, Y):

    kfolds = 5
    f1scores = []

    for i in range(kfolds):
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=i)
        tfidf_vectorizer = TfidfVectorizer()
        X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
        X_test_tfidf = tfidf_vectorizer.transform(X_test)
        clf = LogisticRegression()
        clf.fit(X_train_tfidf, y_train)
        y_pred = clf.predict(X_test_tfidf)
        f1score = f1_score(y_test, y_pred, average='macro')
        f1scores.append(f1score)

    mean_f1score = sum(f1scores) / kfolds

    plt.grid(True)
    plt.plot(range(kfolds), f1scores, marker='o', linestyle='dashed')
    plt.xlabel('Fold')
    plt.ylabel('F1 Score')
    plt.title(f'F1 Scores of {kfolds}-fold Cross Validation')
    plt.xticks(range(kfolds))
    plt.axhline(y=mean_f1score, color='r', linestyle='--', label=f'Average F1 Score {mean_f1score* 100:.2f} %')
    plt.legend()
    plt.show()