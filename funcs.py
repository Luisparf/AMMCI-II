# -*- coding: utf-8 -*-
import re
import string
import nltk
from nltk.corpus import stopwords
import pandas as pd
from sklearn.metrics import f1_score,accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from rake_nltk import Rake
from PIL import Image
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

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
    
    # train_data = ''
    train_data = pd.read_csv('train.csv')
    test_data = pd.read_csv('test.csv')
    # print()
    # print(train_data.head(10))
    # print()

    # train_data = data['selected_text'].apply(lambda x: clean_text(x))
    train_data['selected_text'] = train_data['selected_text'].apply(lambda x: clean_text(x))
    train_data['text'] = train_data['text'].apply(lambda x: clean_text(x))
    test_data['text']  = test_data['text'].apply(lambda x: clean_text(x))
    train_target = train_data['sentiment']
    test_target = test_data['sentiment']
    counts = train_target.value_counts()
    counts1 = test_target.value_counts()
    print(f'\n\nBase de treino:\n{counts}')
    print(f'\nBase de teste:\n{counts1}\n')
   

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
    qualidade das previsões usando a métrica accuracy_score.
    """

    kfolds = 10
    # f1scores_clr = []
    # f1scores_cmnb = []
    # f1scores_crf = []
    accuracies_clr = []
    accuracies_cmnb = []
    accuracies_crf = []


    # Usando TF-IDF
    for i in range(kfolds):
        # X_train, X_test, y_train, y_test = train_test_split(test_data, Y, test_size=0.2, random_state=i)
        
        X_train, _, y_train, _ = train_test_split(train_data, train_target, test_size=0.01, random_state=i)
        _, X_test, _, y_test = train_test_split(test_data, test_target, test_size=0.99, random_state=i)
        tfidf_vectorizer = TfidfVectorizer()
        X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
        X_test_tfidf = tfidf_vectorizer.transform(X_test)

        """Os parâmetros de cada método  são ajustados de acordo com o grid search:"""
        ## Regressão logística
        clr = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=2000)
        clr.fit(X_train_tfidf, y_train)
        y_pred = clr.predict(X_test_tfidf)
        """O f1_score é uma métrica que leva em consideração tanto a precisão quanto o recall do modelo.
        A precisão é a proporção de amostras classificadas corretamente para uma classe específica em relação 
        ao número total de amostras classificadas para essa classe. O recall é a proporção de amostras classificadas corretamente para uma 
        classe específica em relação ao número total de amostras que realmente pertencem a essa classe. O  f1_score é uma média harmônica 
        dessas duas métricas e é uma medida mais robusta para problemas de classificação desbalanceados.
        """
        # f1score_clr = f1_score(y_test, y_pred, average='macro')
        # f1scores_clr.append(f1score_clr)
        # print(f1score_clr)
         ##
        accuracy_clr = accuracy_score(y_test, y_pred)
        print('Accuracy clr:', accuracy_clr)
        accuracies_clr.append(accuracy_clr)
        

        ## Naive Bayes
        cmnb = MultinomialNB(alpha=1.0, fit_prior=False)
        cmnb.fit(X_train_tfidf, y_train)
        y_pred = cmnb.predict(X_test_tfidf)
        # f1score_cmnb = f1_score(y_test, y_pred, average='macro')
        # f1scores_cmnb.append(f1score_cmnb)
        # print(f1score_cmnb)
          ##
        accuracy_cmnb = accuracy_score(y_test, y_pred)
        print('Accuracy cmnb:', accuracy_cmnb)
        accuracies_cmnb.append(accuracy_cmnb)


        ## Random Forest
        crf = RandomForestClassifier(max_depth=None, n_estimators=500)
        crf.fit(X_train_tfidf, y_train)
        y_pred = crf.predict(X_test_tfidf)
        # f1score_crf = f1_score(y_test, y_pred, average='macro')
        # f1scores_crf.append(f1score_crf)
        # print(f1score_crf)
              ##
        accuracy_crf = accuracy_score(y_test, y_pred)
        print('Accuracy crf:', accuracy_crf)
        accuracies_crf.append(accuracy_crf)


    # plt.grid(True)
    # plt.plot(range(kfolds), f1scores_clr, marker='o', linestyle='dashed', label=f'Average LR Score {np.mean(f1scores_clr) * 100:.2f}%', color='purple')
    # plt.plot(range(kfolds), f1scores_cmnb, marker='o', linestyle='dashed', label=f'Average NB Score {np.mean(f1scores_cmnb) * 100:.2f}%', color='red')
    # plt.plot(range(kfolds), f1scores_crf, marker='o', linestyle='dashed', label=f'Average RF Score {np.mean(f1scores_crf) * 100:.2f}%', color='green')
    # plt.xlabel('Fold')
    # plt.ylabel('F1 Score')
    # plt.title(f'F1 Scores of {kfolds}-fold Cross Validation TF-IDF')
    # plt.xticks(range(kfolds))
    # # plt.axhline(y=mean_f1score, color='r', linestyle='-', label=f'Average F1 Score {mean_f1score* 100:.2f} %')
    # plt.legend()
    # plt.savefig('tfidf_f1_final.png')
    # plt.show()

    plt.grid(True)
    plt.plot(range(kfolds), accuracies_clr, marker='o', linestyle='dashed', label=f'Average LR Accuracy {np.mean(accuracies_clr) * 100:.2f}%', color='purple')
    plt.plot(range(kfolds), accuracies_cmnb, marker='o', linestyle='dashed', label=f'Average NB Accuracy {np.mean(accuracies_cmnb) * 100:.2f}%', color='red')
    plt.plot(range(kfolds), accuracies_crf, marker='o', linestyle='dashed', label=f'Average RF Accuracy {np.mean(accuracies_crf) * 100:.2f}%', color='green')
    plt.xlabel('Fold')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracies of {kfolds}-fold Cross Validation TF-IDF')
    plt.xticks(range(kfolds))
    # plt.axhline(y=mean_f1score, color='r', linestyle='-', label=f'Average F1 Score {mean_f1score* 100:.2f} %')
    plt.legend()
    plt.savefig('tfidf_accuracy_final.png')
    plt.show()


######################################################################################################

def rake_tokenizer(text):
    rake = Rake()
    t = rake.extract_keywords_from_text(text)
    return rake.get_ranked_phrases()

######################################################################################################


def run_count(train_data, test_data, train_target, test_target):

  
    kfolds = 10
    # f1scores_clr = []
    # f1scores_cmnb = []
    # f1scores_crf = []
    accuracies_clr = []
    accuracies_cmnb = []
    accuracies_crf = []


    for i in range(kfolds):
        # X_train, _, y_train, _ = train_test_split(train_data, Y, test_size=0.2, random_state=i)
        # _, X_test, _, y_test = train_test_split(test_data, Y, test_size=0.2, random_state=i)
        # X_train, X_test, y_train, y_test = train_test_split(test_data, Y, test_size=0.2, random_state=i)
        X_train, _, y_train, _ = train_test_split(train_data, train_target, test_size=0.01, random_state=i)
        _, X_test, _, y_test = train_test_split(test_data, test_target, test_size=0.99, random_state=i)

              
        vectorizer = CountVectorizer()
        X_train_rake = vectorizer.fit_transform(X_train)
        X_test_rake = vectorizer.transform(X_test)

        """
        vectorizer = CountVectorizer() - cria um objeto CountVectorizer para converter as palavras em um vetor de contagem de palavras.

        X_train_rake = vectorizer.fit_transform(X_train) - ajusta o vetorizador ao conjunto de treinamento X_train e 
        transforma o conjunto de treinamento em um vetor de contagem de palavras. Isso significa que o vetorizador aprende o vocabulário 
        do conjunto de treinamento e cria um vetor de contagem 
        """

      
        """Os parâmetros de cada método  são ajustados de acordo com o grid search:"""
        ## Regressão logística
        clr = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=2000)
        clr.fit(X_train_rake, y_train)
        y_pred = clr.predict(X_test_rake)
        # f1score_clr = f1_score(y_test, y_pred, average='macro')
        # f1scores_clr.append(f1score_clr)
        # print(f1score_clr)
        ##
        accuracy_clr = accuracy_score(y_test, y_pred)
        print('Accuracy clr:', accuracy_clr)
        accuracies_clr.append(accuracy_clr)

        ## Naive Bayes
        cmnb = MultinomialNB(alpha=1.0, fit_prior=False )
        cmnb.fit(X_train_rake, y_train)
        y_pred = cmnb.predict(X_test_rake)
        # f1score_cmnb = f1_score(y_test, y_pred, average='macro')
        # f1scores_cmnb.append(f1score_cmnb)
        # print(f1score_cmnb)
         ##
        accuracy_cmnb = accuracy_score(y_test, y_pred)
        print('Accuracy cmnb:', accuracy_cmnb)
        accuracies_cmnb.append(accuracy_cmnb)


        ## Random Forest
        crf = RandomForestClassifier(max_depth=None, n_estimators=500)
        crf.fit(X_train_rake, y_train)
        y_pred = crf.predict(X_test_rake)
        # f1score_crf = f1_score(y_test, y_pred, average='macro')
        # f1scores_crf.append(f1score_crf)
        # print(f1score_crf)
            ##
        accuracy_crf = accuracy_score(y_test, y_pred)
        print('Accuracy crf:', accuracy_crf)
        accuracies_crf.append(accuracy_crf)


    # plt.grid(True)
    # plt.plot(range(kfolds), f1scores_clr, marker='o', linestyle='dashed', label=f'Average LR Score {np.mean(f1scores_clr) * 100:.2f}%', color='purple')
    # plt.plot(range(kfolds), f1scores_cmnb, marker='o', linestyle='dashed', label=f'Average NB Score {np.mean(f1scores_cmnb) * 100:.2f}%', color='red')
    # plt.plot(range(kfolds), f1scores_crf, marker='o', linestyle='dashed', label=f'Average RF Score {np.mean(f1scores_crf) * 100:.2f}%', color='green')
    # plt.xlabel('Fold')
    # plt.ylabel('F1 Score')
    # plt.title(f'F1 Scores of {kfolds}-fold Cross Validation CountVectorizer')
    # plt.xticks(range(kfolds))
    # # plt.axhline(y=mean_f1score, color='r', linestyle='-', label=f'Average F1 Score {mean_f1score* 100:.2f} %')
    # plt.legend()
    # plt.savefig('count_f1_final.png')
    # plt.show()

    plt.grid(True)
    plt.plot(range(kfolds), accuracies_clr, marker='o', linestyle='dashed', label=f'Average LR Accuracy {np.mean(accuracies_clr) * 100:.2f}%', color='purple')
    plt.plot(range(kfolds), accuracies_cmnb, marker='o', linestyle='dashed', label=f'Average NB Accuracy {np.mean(accuracies_cmnb) * 100:.2f}%', color='red')
    plt.plot(range(kfolds), accuracies_crf, marker='o', linestyle='dashed', label=f'Average RF Accuracy {np.mean(accuracies_crf) * 100:.2f}%', color='green')
    plt.xlabel('Fold')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracies of {kfolds}-fold Cross Validation CountVectorizer')
    plt.xticks(range(kfolds))
    # plt.axhline(y=mean_f1score, color='r', linestyle='-', label=f'Average F1 Score {mean_f1score* 100:.2f} %')
    plt.legend()
    plt.savefig('count_accuracy_final.png')
    plt.show()


######################################################################################################

def sentiment_score():
    """SentimentIntensityAnalyzer é uma classe da biblioteca NLTK (Natural Language Toolkit) que é usada para analisar o sentimento de textos em inglês. 
    Ele usa um modelo pré-treinado para calcular uma pontuação de sentimentos que varia de -1 a 1, onde um valor mais próximo de -1 indica um 
    sentimento negativo e um valor mais próximo de 1 indica um sentimento positivo.

     O SentimentIntensityAnalyzer usa um modelo de análise de sentimento baseado em regras, que é treinado em uma ampla gama de dados, 
     incluindo análises de produtos, críticas de filmes e tweets. O modelo usa uma abordagem de análise léxica para atribuir uma pontuação 
     de sentimento a cada palavra no texto e, em seguida, calcula uma pontuação agregada com base em todas as palavras no texto.

    Além da pontuação geral de sentimento, o SentimentIntensityAnalyzer também fornece pontuações individuais para os 
    sentimentos positivo, negativo, neutro e misto, bem como uma lista de tokens (palavras) com suas respectivas pontuações de sentimento.
      Essas informações podem ser usadas para uma análise mais detalhada do sentimento do texto."""

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

######################################################################################################

def describe():
    
    # Load the data into a Pandas DataFrame
    data = pd.read_csv('test.csv')

    # Clean the 'text' column by applying the 'clean_text' function
    data['text'] = data['text'].apply(lambda x: clean_text(x))

    # Remove rows with empty 'text' values
    data = data[data['text'] != '']

    # Calculate descriptive statistics for the 'text' column
    text_stats = data['text'].describe()

    # Calculate descriptive statistics for the 'sentiment' column
    sentiment_stats = data['sentiment'].describe()

    # Print the results
    print("Text Statistics:\n", text_stats)
    print("\nSentiment Statistics:\n", sentiment_stats)

######################################################################################################

def grid_search(train_data, test_data, train_target, test_target):

    X_train, _, y_train, _ = train_test_split(train_data, train_target, test_size=0.01, random_state=5)
    # _, X_test, _, y_test = train_test_split(test_data, test_target, test_size=0.99, random_state=5)

    models = [
        ('lr', LogisticRegression(max_iter=4000), {
            'clf__penalty': [ 'l2', 'none'],
            'clf__solver': ['lbfgs', 'sag', 'saga']
        }),
        ('nb', MultinomialNB(), {
            'clf__alpha': [0.1, 1.0, 10.0],
            'clf__fit_prior': [True, False]
        }),
        ('rf', RandomForestClassifier(), {
            'clf__n_estimators': [10, 100, 500],
            'clf__max_depth': [10, 50, None]
        })
    ]

    # Itera por cada modelo e faz o grid search
    for name, model, params in models:
        # Cria um pipeline para vetorizar o texto e treina o modelo
        pipe = Pipeline([
            ('vect', TfidfVectorizer()),
            ('clf', model)
        ])

        # Define os hiperparâmetros para a busca
        parameters = {
            'vect__ngram_range': [(1,1), (1,2)], 
            **params
        }

        # Usa o GridSearchCV para buscar o melhor hiperparâmetro
        grid_search = GridSearchCV(pipe, parameters, cv=5, n_jobs=-1, verbose=1)

        # Ajusta o objeto GridSearchCV  para o treinamento:
        grid_search.fit(X_train, y_train)

        # Mostra o melhor parâmetro com o melhor score
        print(f'\n\nBest score ({name}):', grid_search.best_score_)
        print(f'Best parameters ({name}):\n\n', grid_search.best_params_)
