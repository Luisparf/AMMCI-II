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
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from rake_nltk import Rake
# import RAKE
from sklearn.feature_extraction.text import CountVectorizer

import matplotlib.pyplot as plt

# Pré processamento #################################################################################

def clean_text(text):
    # Remove pontuação
    
    if isinstance(text, str) and not pd.isnull(text) and not None: # se o texto é número ou nulo, ignora

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
    target = data['sentiment']
    return train_data, text_data, target

######################################################################################################

def run_tfidf(train_data, test_data, Y):

    kfolds = 10
    f1scores_clr = []
    f1scores_cmnb = []
    f1scores_crf = []


    # Usando TF-IDF
    for i in range(kfolds):
        # X_train, _, y_train, _ = train_test_split(train_data, Y, test_size=0.2, random_state=i)
        X_train, X_test, y_train, y_test = train_test_split(train_data, Y, test_size=0.2, random_state=i)
        # _, X_test, _, y_test = train_test_split(test_data, Y, test_size=0.2, random_state=i)
        tfidf_vectorizer = TfidfVectorizer()
        X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
        X_test_tfidf = tfidf_vectorizer.transform(X_test)
        
        ## Regressão logística
        clr = LogisticRegression()
        clr.fit(X_train_tfidf, y_train)
        y_pred = clr.predict(X_test_tfidf)
        f1score_clr = f1_score(y_test, y_pred, average='macro')
        f1scores_clr.append(f1score_clr)
        print(f1score_clr)

        ## Naive Bayes
        cmnb = MultinomialNB()
        cmnb.fit(X_train_tfidf, y_train)
        y_pred = cmnb.predict(X_test_tfidf)
        f1score_cmnb = f1_score(y_test, y_pred, average='macro')
        f1scores_cmnb.append(f1score_cmnb)
        print(f1score_cmnb)

        ## Random Forest
        crf = RandomForestClassifier(n_estimators=100)
        crf.fit(X_train_tfidf, y_train)
        y_pred = crf.predict(X_test_tfidf)
        f1score_crf = f1_score(y_test, y_pred, average='macro')
        f1scores_crf.append(f1score_crf)
        print(f1score_crf)

    plt.grid(True)
    plt.plot(range(kfolds), f1scores_clr, marker='o', linestyle='dashed', label=f'Average LR Score {f1score_clr * 100:.2f}%', color='purple')
    plt.plot(range(kfolds), f1scores_cmnb, marker='o', linestyle='dashed', label=f'Average NB Score {f1score_cmnb * 100:.2f}%', color='red')
    plt.plot(range(kfolds), f1scores_crf, marker='o', linestyle='dashed', label=f'Average RF Score {f1score_crf * 100:.2f}%', color='green')
    plt.xlabel('Fold')
    plt.ylabel('F1 Score')
    plt.title(f'F1 Scores of {kfolds}-fold Cross Validation with TF-IDF')
    plt.xticks(range(kfolds))
    # plt.axhline(y=mean_f1score, color='r', linestyle='-', label=f'Average F1 Score {mean_f1score* 100:.2f} %')
    plt.legend()
    # for i, score in enumerate(f1scores_clr):
    #     plt.annotate(f'{score * 100:.2f}', xy=(i, score), xytext=(-10, 5), textcoords='offset points',ha='center', va='bottom')
    # for i, j in zip(range(kfolds), f1scores_clr):
    #     plt.annotate(f'{j * 100:.2f}%', xy=(i,j), xytext =(0.1 * offset, -offset * 0.5),  
    #         textcoords ='offset points',ha='center', va='bottom')
    plt.savefig('tf-idf.png')

    plt.show()

######################################################################################################

def rake_tokenizer(text):
    rake = Rake()
    rake.extract_keywords_from_text(text)
    return rake.get_ranked_phrases()

######################################################################################################


def run_rake(train_data, test_data, Y):

    kfolds = 10

    f1scores_clr = []
    f1scores_cmnb = []
    f1scores_crf = []

    for i in range(kfolds):
        X_train, X_test, y_train, y_test = train_test_split(train_data, Y, test_size=0.2, random_state=i)

        # Vectorize the text data using RAKE
        # rake = Rake()
        vectorizer = CountVectorizer(tokenizer=rake_tokenizer)
        X_train_rake = vectorizer.fit_transform(X_train)
        X_test_rake = vectorizer.transform(X_test)

        # Train a logistic regression model on the vectorized data
      
      
        ## Regressão logística
        clr = LogisticRegression()
        clr.fit(X_train_rake, y_train)
        y_pred = clr.predict(X_test_rake)
        f1score_clr = f1_score(y_test, y_pred, average='macro')
        f1scores_clr.append(f1score_clr)
        print(f1score_clr)

        ## Naive Bayes
        cmnb = MultinomialNB()
        cmnb.fit(X_train_rake, y_train)
        y_pred = cmnb.predict(X_test_rake)
        f1score_cmnb = f1_score(y_test, y_pred, average='macro')
        f1scores_cmnb.append(f1score_cmnb)
        print(f1score_cmnb)

        ## Random Forest
        crf = RandomForestClassifier(n_estimators=100)
        crf.fit(X_train_rake, y_train)
        y_pred = crf.predict(X_test_rake)
        f1score_crf = f1_score(y_test, y_pred, average='macro')
        f1scores_crf.append(f1score_crf)
        print(f1score_crf)



    # mean_f1score = sum(f1scores_clr) / kfolds
    # offset = 10


    plt.grid(True)
    plt.plot(range(kfolds), f1scores_clr, marker='o', linestyle='dashed', label=f'Average LR Score {f1score_clr * 100:.2f}%', color='purple')
    plt.plot(range(kfolds), f1scores_cmnb, marker='o', linestyle='dashed', label=f'Average NB Score {f1score_cmnb * 100:.2f}%', color='red')
    plt.plot(range(kfolds), f1scores_crf, marker='o', linestyle='dashed', label=f'Average RF Score {f1score_crf * 100:.2f}%', color='green')
    plt.xlabel('Fold')
    plt.ylabel('F1 Score')
    plt.title(f'F1 Scores of {kfolds}-fold Cross Validation with RAKE')
    plt.xticks(range(kfolds))
    # plt.axhline(y=mean_f1score, color='r', linestyle='-', label=f'Average F1 Score {mean_f1score* 100:.2f} %')
    plt.legend()
    plt.savefig('rake.png')

    plt.show()

