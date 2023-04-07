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
from PIL import Image
import numpy as np

# import RAKE
from sklearn.feature_extraction.text import CountVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
# nltk.download('vader_lexicon')
# from wordcloud import WordCloud, STOPWORDS


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
    
    train_data = ''
    train_data = pd.read_csv('train.csv')
    test_data = pd.read_csv('test.csv')
    # print(data.head(10))
  
    # train_data = data['selected_text'].apply(lambda x: clean_text(x))
    train_data['selected_text'] = train_data['selected_text'].apply(lambda x: clean_text(x))
    train_data['text'] = train_data['text'].apply(lambda x: clean_text(x))
    test_data['text']  = test_data['text'].apply(lambda x: clean_text(x))
    train_target = train_data['sentiment']
    test_target = test_data['sentiment']
    test_data = test_data['text']
    train_data['features'] = train_data['selected_text'] + ' ' + train_data['text']

    return train_data['features'], test_data, train_target, test_target

######################################################################################################

def run_tfidf(train_data, test_data, train_target, test_target):
    
    """
    O primeiro passo é dividir o conjunto de dados em conjuntos de treinamento e teste, usando a função "train_test_split" da biblioteca "sklearn". 
    O parâmetro "test_size" define a proporção do conjunto de teste em relação ao conjunto de dados total e "random_state" é usado para controlar a 
        aleatoriedade da divisão de dados e garantir que os resultados sejam reproduzíveis.

    Em seguida, o vetorizador Tf-idf é criado usando a função "TfidfVectorizer()" e é usado para transformar o conjunto de treinamento em um conjunto de vetores Tf-idf. 
    Esse processo envolve a contagem de frequência das palavras em cada documento, a ponderação dessa frequência pela frequência inversa do documento em que a palavra aparece, 
    e a normalização dos resultados.

    O conjunto de teste também é transformado em vetores Tf-idf usando o mesmo vetorizador criado anteriormente. 
    Isso é feito para garantir que os dados de teste sejam representados da mesma forma que os dados de treinamento.

    Em seguida, os modelos de Regressão Logística, Naive Bayes e Random Forest são criados e treinados com os dados de treinamento usando a função "fit()" das classes "LogisticRegression()", 
    "MultinomialNB()" e "RandomForestClassifier()". 
    Depois que o modelo é treinado, ele é usado para prever as classes do conjunto de teste usando a função "predict()".

    Finalmente, a métrica F1-score é calculada para avaliar a qualidade das previsões. O parâmetro "average" é definido como "macro" para calcular o F1-score médio ponderado pelas classes. 
    Isso é importante porque, em um problema de classificação multiclasse, algumas classes podem ser menos frequentes do que outras e, portanto, têm menos influência no F1-score geral.

    Em resumo, esse código é um exemplo de como utilizar a técnica de Regressão Logística e a técnica de vetorização de texto com Tf-idf para classificar dados de texto e avaliar a 
    qualidade das previsões usando a métrica F1-score.
    """

    kfolds = 10
    f1scores_clr = []
    f1scores_cmnb = []
    f1scores_crf = []


    # Usando TF-IDF
    for i in range(kfolds):
        # X_train, X_test, y_train, y_test = train_test_split(test_data, Y, test_size=0.2, random_state=i)
        
        X_train, _, y_train, _ = train_test_split(train_data, train_target, test_size=0.1, random_state=i)
        _, X_test, _, y_test = train_test_split(test_data, test_target, test_size=0.1, random_state=i)
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
    plt.savefig('tf-idf_final.png')

    plt.show()

######################################################################################################

def rake_tokenizer(text):
    rake = Rake()
    rake.extract_keywords_from_text(text)
    return rake.get_ranked_phrases()

######################################################################################################


def run_rake(train_data, test_data, train_target, test_target):

  
    """
    A função "rake_tokenizer" utiliza a biblioteca RAKE para extrair as frases-chave de um texto. 
    RAKE é uma técnica de extração de palavras-chave que considera a frequência das palavras e a coocorrência entre elas para identificar as frases mais importantes de um texto. 
    A função "tokenizer" da classe "CountVectorizer" é utilizada para passar o texto através da função "rake_tokenizer" e gerar uma lista de frases-chave.

    Em seguida, o objeto "CountVectorizer" é criado com a opção "tokenizer=rake_tokenizer" para vetorizar as frases-chave extraídas pelo RAKE.
    O conjunto de treinamento é transformado em um conjunto de vetores usando a função "fit_transform()" do objeto "CountVectorizer", 
    e o conjunto de teste é transformado em vetores usando a função "transform()".

    Depois disso, dois modelos de classificação são aplicados nos conjuntos de vetores gerados. O primeiro modelo é uma Regressão Logística, criado com a classe "LogisticRegression()". 
    O modelo é treinado com os dados de treinamento utilizando a função "fit()", e as previsões são feitas para o conjunto de teste utilizando a função "predict()". 
    A métrica F1-score é calculada utilizando a função "f1_score()" para avaliar a qualidade das previsões.

    O segundo modelo é um Naive Bayes Multinomial, criado com a classe "MultinomialNB()". O modelo é treinado da mesma forma que a Regressão Logística, 
    e as previsões são feitas utilizando a função "predict()".

    Finalmente, o código apresenta o valor da métrica F1-score para o modelo de Regressão Logística utilizando a técnica RAKE para vetorização de texto. 
     É importante notar que esse código é apenas um exemplo e que outras técnicas de vetorização e modelos de classificação podem ser utilizados dependendo do problema em questão.
    """

    kfolds = 10
    f1scores_clr = []
    f1scores_cmnb = []
    f1scores_crf = []

    for i in range(kfolds):
        # X_train, _, y_train, _ = train_test_split(train_data, Y, test_size=0.2, random_state=i)
        # _, X_test, _, y_test = train_test_split(test_data, Y, test_size=0.2, random_state=i)
        # X_train, X_test, y_train, y_test = train_test_split(test_data, Y, test_size=0.2, random_state=i)
        X_train, _, y_train, _ = train_test_split(train_data, train_target, test_size=0.1, random_state=i)
        _, X_test, _, y_test = train_test_split(test_data, test_target, test_size=0.1, random_state=i)

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
    plt.savefig('rake_final.png')

    plt.show()

######################################################################################################

def sentiment_score():

    sentiments = SentimentIntensityAnalyzer()

    data = pd.read_csv('train.csv')
    data['text'] = data['text'].apply(lambda x: clean_text(x))
    # data['selected_text'] = data['text'].apply(lambda x: clean_text(x))

    data["Positive"] = [sentiments.polarity_scores(i)["pos"] for i in data["text"]]
    data["Negative"] = [sentiments.polarity_scores(i)["neg"] for i in data["text"]]
    data["Neutral"] = [sentiments.polarity_scores(i)["neu"] for i in data["text"]]
    data = data[["text", "Positive", 
             "Negative", "Neutral"]]
    print(data.head(30))

######################################################################################################

def word_clo():

    data = pd.read_csv('train.csv')
    data['text'] = data['text'].apply(lambda x: clean_text(x))
    data['selected_text'] = data['selected_text'].apply(lambda x: clean_text(x))
    data = pd.concat([data['selected_text'], data['text']], axis=1)


    twitter_mask = np.array(Image.open("test.png"))

    text = " ".join(i for i in data.text)

    stopwords = set(STOPWORDS)

    wordcloud = WordCloud(stopwords=stopwords, background_color="black", mask=twitter_mask, width=1600, height=800, max_words=2000, 
                          min_font_size=1, max_font_size=200).generate(text)

    plt.figure( figsize=(16,8))

    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()