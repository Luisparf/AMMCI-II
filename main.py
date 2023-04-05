# -*- coding: utf-8 -*-

# import pandas as pd
# import numpy as np

# nltk.download('stopwords')
# nltk.download('punkt')
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import f1_score
# from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt

from funcs import  load_data, k_folds


if __name__ == '__main__':
    
    train_data, text_data, train_target = load_data()
    
    # train_features = data.drop( 'textID', axis=1)
    # train_features = data.drop(['sentiment', 'textID'], axis=1)


    # print(train_features)
    # train_features = train_features.drop('species', axis=1)]

    # print(f"text  = {data['text'].shape}")
    # print(f"selected text = {data['selected_text'].shape}")
    # print(f"sentiment = {data['sentiment'].shape}")
    # print(f"text/selected = {data[['textID', 'selected_text']].shape(axis=1)}")



    # # Split the data into train and test sets
    # X_train,_, y_train , _ = train_test_split(train_data, train_target, test_size=0.2, random_state=42)
    # _ ,X_test, _,y_test = train_test_split(text_data, train_target, test_size=0.2, random_state=42)
    # X_train,X_test, y_train ,y_test  = train_test_split(data['text'], train_target, test_size=0.2, random_state=42)

    # print(f'X_train = {X_train}\n\n')
    # print(f'y_train = {y_train}\n\n')
    # print(f'X_test = {X_test}\n\n')
    # print(f'y_test = {y_test}\n\n')

    # for x in X_train:
    #     print(x)

    # # Vectorize the text data using TF-IDF
    # tfidf_vectorizer = TfidfVectorizer()
    # X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    # X_test_tfidf = tfidf_vectorizer.transform(X_test)

    # # # Train a logistic regression model on the vectorized data
    # clf = LogisticRegression()
    # clf.fit(X_train_tfidf, y_train)

    # # # Evaluate the model using F1 score
    # y_pred = clf.predict(X_test_tfidf)
    # f1score = f1_score(y_test, y_pred, average='macro')
    # print('F1 score:', f1score)

    # k_folds(text_data,train_target)
    k_folds(train_data,train_target)


   