# -*- coding: utf-8 -*-
import re
import string
import nltk
from nltk.corpus import stopwords
import pandas as pd

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
