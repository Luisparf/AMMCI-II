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

from funcs import  load_data, run_tfidf, run_rake, sentiment_score, word_clo
import sys

if __name__ == '__main__':
    
    train_data, test_data, train_target, test_target = load_data()

    # print(train_data_select_text)
    # print(train_data_text)
    # print(test_data_text)
    # print(test_target)
    # sentiment_score()
    # word_clo()
    # print(sys.executable)

    
    run_tfidf(train_data, test_data, train_target, test_target)
    run_rake(train_data, test_data, train_target, test_target)


   