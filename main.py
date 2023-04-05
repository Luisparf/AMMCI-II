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

from funcs import  load_data, run_tfidf, run_rake


if __name__ == '__main__':
    
    train_data, test_data, target = load_data()
    
    run_tfidf(train_data,test_data,target)
    run_rake(train_data,test_data,target)


   