import pandas as pd
import numpy as np
import nltk
import sklearn
import operator
import os
import zipfile
import io
from sklearn.linear_model import LogisticRegression

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('omw-1.4')
nltk.download('wordnet')

# functions
def load_data():
    data_path = "../coursework/bbc"
    categories = ['business', 'entertainment', 'politics', 'sport', 'tech']
    news_list = []
    labels = []
    for category in categories:
        category_path = os.path.join(data_path, category)
        for new in os.listdir(category_path):
            if new.endswith('.txt'):
                with open(os.path.join(category_path, new), encoding='utf-8', errors='ignore') as f:
                    news_list.append(f.read())
                    labels.append(category)
    return news_list, labels

def get_word_list_in_sentence(string):
    lemmatizer = nltk.stem.WordNetLemmatizer()
    sentence_split = nltk.tokenize.sent_tokenize(string)
    word_list = []
    for sentence in sentence_split:
        pre_word_list = nltk.tokenize.word_tokenize(sentence)
        for word in pre_word_list:
            word_list.append(lemmatizer.lemmatize(word).lower())
    return word_list

def create_dictionary(pos_dataset, neg_dataset, max_num_features):
    # Splitting words, eliminating deactivated words, counting frequencies
    word_count_list = {}
    stopwords = set(nltk.corpus.stopwords.words('english'))
    stopwords.add('.')
    stopwords.add(',')
    stopwords.add("--")
    stopwords.add("``")
    for pos_review in pos_dataset:
        word_list = get_word_list_in_sentence(pos_review)
        for word in word_list:
            if word in stopwords:
                continue
            elif word in word_count_list:
                word_count_list[word] += 1
            else:
                word_count_list[word] = 1
    for neg_review in neg_dataset:
        word_list = get_word_list_in_sentence(neg_review)
        for word in word_list:
            if word in stopwords:
                continue
            elif word in word_count_list:
                word_count_list[word] += 1
            else: word_count_list[word] = 1
    # Sorting, creating dictionaries
    sorted_list = sorted(word_count_list.items(), key=operator.itemgetter(1), reverse=True)[:max_num_features]
    dictionary = []
    for word, frequency in sorted_list:
        dictionary.append(word)
    return dictionary

def get_word_vector(vocabulary, string):
    word_vector = np.zeros(len(vocabulary))
    word_list = get_word_list_in_sentence(string)
    for index, word in enumerate(vocabulary):
        if word in word_list:
            word_vector[index] = word_list.count(word)  # Count the number of times the word appears in the sentence
    return word_vector

def log_reg_muti_training(news_list, labels, dictionary):
    # Building the training set
    X_train = []
    Y_train = []
    for index, new in enumerate(news_list):
        new_vector = get_word_vector(dictionary, new)
        X_train.append(new_vector)
        if labels[index] == 'business':
            Y_train.append(0)
        elif labels[index] == 'entertainment':
            Y_train.append(1)
        elif labels[index] == 'politics':
            Y_train.append(2)
        elif labels[index] == 'sport':
            Y_train.append(3)
        elif labels[index] == 'tech':
            Y_train.append(4)
    X_train = np.asarray(X_train)
    Y_train = np.asarray(Y_train)
    # training
    log_reg_muti = LogisticRegression(multi_class="multinomial", solver='lbfgs', C=10, random_state=42)
    log_reg_muti.fit(X_train, Y_train)
    return log_reg_muti


# running
news_list, labels = load_data()
"""
for new in news_list[510:516]:
    print(new)
for category in labels[510:516]:
    print(category)
"""

dictionary = create_dictionary(news_list, labels, 2000)
log_reg_muti = log_reg_muti_training(news_list, labels, dictionary)

