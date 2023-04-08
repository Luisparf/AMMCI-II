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

from funcs import  load_data, run_tfidf, run_count, sentiment_score, word_clo, describe, grid_search
import sys

if __name__ == '__main__':
    
    train_data, test_data, train_target, test_target = load_data()

    # grid_search(train_data, test_data, train_target, test_target)

    # print(train_data_select_text)
    # print(train_data_text)
    # print(test_data_text)
    # print(test_target)
    ## Mostra um score de sentimento dos 30 primeiros tweets:
    # sentiment_score()
    ## Cria e mostra o word cloud:
    # word_clo()
    # print(sys.executable)
    # describe()

    """
    TfidfVectorizer e CountVectorizer são duas técnicas populares para a vetorização de texto, 
    que é o processo de transformar um texto em um vetor numérico para que possa ser usado em algoritmos de aprendizado de máquina.

    A principal diferença entre essas duas técnicas é que o CountVectorizer simplesmente conta a frequência das palavras em cada documento, 
    enquanto o TfidfVectorizer considera não apenas a frequência das palavras em um documento, mas também a frequência da palavra em todo o corpus.

    O TfidfVectorizer também leva em consideração a raridade das palavras, dando mais peso a palavras que aparecem com menos frequência em todo o corpus, 
    o que geralmente as torna mais informativas.

    Em termos de semelhanças, ambas as técnicas são úteis para transformar texto em uma representação numérica q
    ue pode ser usada em algoritmos de aprendizado de máquina. Ambas também são capazes de lidar com grandes quantidades de dados de texto e são relativamente fáceis de usar.

    Ambas também possuem parâmetros que podem ser ajustados, como a escolha da frequência mínima de uma palavra para ser considerada na vetorização, 
    a escolha do tipo de n-grama a ser considerado (uni-gramas, bi-gramas, tri-gramas, etc.), e o tipo de tokenização a ser usado.

    Em resumo, a principal diferença entre TfidfVectorizer e CountVectorizer é que o primeiro leva em conta a raridade das palavras, enquanto o segundo não.
      No entanto, ambas as técnicas têm suas vantagens e desvantagens, e a escolha entre elas depende dos objetivos do projeto e 
      do tipo de dados de texto que está sendo trabalhado.
    """
    # run_tfidf(train_data, test_data, train_target, test_target)
    # run_count(train_data, test_data, train_target, test_target)


   