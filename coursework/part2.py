import re
import numpy as np
import nltk
import operator
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score,recall_score,f1_score,accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import gensim.downloader as api
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('omw-1.4')
nltk.download('wordnet')

embeddings_dictionary = api.load("glove-wiki-gigaword-100")

# 1 functions
def load_data():
    """
    Function: Load all news datasets
    :return: full_dataset: A list where each element is a tuple, the first item is the content of the news article, and the second item is the classification result of the news article.
    """
    data_path = "../coursework/bbc"
    categories = [('business', 0), ('entertainment', 1), ('politics', 2), ('sport', 3), ('tech', 4)]
    full_dataset = []
    for category in categories:
        category_path = os.path.join(data_path, category[0])
        for new in os.listdir(category_path):
            if new.endswith('.txt'):
                with open(os.path.join(category_path, new), encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    full_dataset.append((content, category[1]))
    return full_dataset

def get_word_list_in_sentence(string):
    """
    Function: Natural Language Preprocessing
    :param string: A news article.
    :return: Individual word lists after word splitting, lowercase transfer, removing non-literal information, and removing deactivated words.
    """
    lemmatizer = nltk.stem.WordNetLemmatizer()
    sentence_split = nltk.tokenize.sent_tokenize(string)
    word_list = []
    for sentence in sentence_split:
        pre_word_list = re.findall(r'\b\w+\b', sentence.lower())
        for word in pre_word_list:
            word_list.append(lemmatizer.lemmatize(word))
    return word_list

def create_frequency_dictionary(full_dataset, max_num_features):
    """
    Function: Creating a Global Dictionary.
    :param full_dataset: Global dataset.
    :param max_num_features: Maximum number of words in the frequency dictionary.
    :return: Frequency dictionary.
    """
    # Splitting words, eliminating deactivated words, counting frequencies
    word_count_list = {}
    stopwords = set(nltk.corpus.stopwords.words('english'))
    stopwords.update(['.', ',', "--", "``", ":", "\n", "'", '"', "''", "`", "´", "-", "—"])

    for new_and_category in full_dataset:
        word_list = get_word_list_in_sentence(new_and_category[0])
        for word in word_list:
            if word in stopwords:
                continue
            elif word in word_count_list:
                word_count_list[word] += 1
            else:
                word_count_list[word] = 1
    # Sorting, creating dictionaries
    sorted_list = sorted(word_count_list.items(), key=operator.itemgetter(1), reverse=True)[:max_num_features]
    dictionary = []
    for word, frequency in sorted_list:
        dictionary.append(word)
    return dictionary

def get_frequency_vector(dictionary, string):
    """
    Function: Processing inputs according to the frequency dictionary.
    :param dictionary: Frequency dictionary.
    :param string: The text string to be processed.
    :return: Processed text vector
    """
    word_vector = np.zeros(len(dictionary))
    word_list = get_word_list_in_sentence(string)
    for index, word in enumerate(dictionary):
        if word in word_list:
            word_vector[index] = word_list.count(word)  # Count the number of times the word appears in the sentence
    return word_vector

def get_embedding_vector(embeddings_dictionary, string):
    """
    Compute the word embedding feature vector for the given text.
    :param string: String of text.
    :param embeddings_dictionary: A dictionary that maps words to their coaddrresponding word embedding vectors.
    :return: Word embedding feature vector for a given text.
    """
    word_list = get_word_list_in_sentence(string)
    embedding_vector = []
    for word in word_list:
        if word in embeddings_dictionary and not np.all(embeddings_dictionary[word] == 0):
            embedding_vector.append(embeddings_dictionary[word])
    if embedding_vector:
        embedding_vector = np.mean(embedding_vector, axis=0)
    else:
        embedding_vector = np.zeros_like(next(iter(embeddings_dictionary.keys())))
    if np.isnan(embedding_vector).any():
        embedding_vector = np.zeros_like(embedding_vector)
    return embedding_vector

def get_combined_vector(frequency_dictionary, embeddings_dictionary, dataset):
    """
    Combine multiple features in a given dataset.
    :param frequency_dictionary: Frequency_dictionary.
    :param embeddings_dictionary: Embeddings_dictionary.
    :param dataset: The given dataset.
    :return:
    """
    X = []
    Y = []
    for new in dataset:
        frequency_vector = get_frequency_vector(frequency_dictionary, new[0])
        embeddings_vector = get_embedding_vector(embeddings_dictionary, new[0])
        combined_vector = np.concatenate((frequency_vector, embeddings_vector))
        X.append(combined_vector)
        Y.append(new[1])
    X = np.asarray(X)
    Y = np.asarray(Y)
    return X, Y

def log_reg_muti_training(dataset, frequency_dictionary, embeddings_dictionary):
    """
    Function: Training logistic regression models
    :param dataset: Specified data set.
    :param frequency_dictionary: Frequency dictionary.
    :return: Well-trained logistic regression classifiers.
    """
    # Building the training set
    X, Y = get_combined_vector(frequency_dictionary, embeddings_dictionary, dataset)
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=0.95)),
        ("logistic", LogisticRegression(multi_class="multinomial", solver='lbfgs', C=10, random_state=42))
    ])
    pipeline.fit(X, Y)
    return pipeline

def kfold_training(training_and_dev_set, k):
    """
    Function: k-fold cross-validation, and tuning parameter by grid search.
    :param training_and_dev_set: The training and development sets of data.
    :param k: The k value of k-fold.
    :return: The best classifier with the best parameters is selected after grid search, and its frequency dictionary.
    """
    best_model = None
    highest_accuracy = 0
    best_frequency_dictionary = []
    # Create a k-fold cross-validator and set k
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    # Directly generate a list of indexes for all k-fold cases of training and test set data.
    all_k_index_list_of_training_set_and_dev_set = kfold.split(training_and_dev_set)
    # Training and verifying by each k of iterations
    accuracy_all = 0
    loop_count = 0
    for this_k_training_set_index_list, this_k_dev_set_index_list in all_k_index_list_of_training_set_and_dev_set:
        training_set = []
        dev_set = []
        # (1) Create a list of training and test sets under this k
        for index, new in enumerate(training_and_dev_set):
            if index in this_k_training_set_index_list:
                training_set.append(new)
            else:
                dev_set.append(new)
        # (2) Training the model under this k
        # Create a global dictionary of reserved keywords under this k
        frequency_dictionary = create_frequency_dictionary(training_set, 2000)
        # Training the model under this k
        kfold_log_reg_muti_model = log_reg_muti_training(training_set, frequency_dictionary, embeddings_dictionary)

        # (3) Use accuracy to validate the performance of the model under this k
        X_dev, Y_dev_gold = get_combined_vector(frequency_dictionary, embeddings_dictionary, dev_set)
        Y_dev_predictions = kfold_log_reg_muti_model.predict(X_dev)
        accuracy = accuracy_score(Y_dev_gold, Y_dev_predictions)
        if accuracy > highest_accuracy:
            best_model = kfold_log_reg_muti_model
            highest_accuracy = accuracy
            best_frequency_dictionary = frequency_dictionary
        loop_count += 1
        print(f"The accuracy in {loop_count} training is: {round(accuracy, 3)}")
        print("======================================================================================")
        print(frequency_dictionary[:50])
        accuracy_all += accuracy
    # Find the average accuracy
    accuracy_average = round(accuracy_all / k, 3)
    print(f"The average accuracy in k-fold is: {round(accuracy_average, 3)}")
    print(f"Highest accuracy in k-fold is {round(highest_accuracy, 3)}")
    return best_model, best_frequency_dictionary


# 2 running
full_dataset = load_data()

training_and_dev_set, test_set = train_test_split(full_dataset, test_size=0.2, random_state=42, shuffle=True)
clf_model, best_frequency_dictionary= kfold_training(training_and_dev_set, 5)

# 3 test
X_test, Y_test_gold = get_combined_vector(best_frequency_dictionary, embeddings_dictionary, test_set)
Y_test_predictions = clf_model.predict(X_test)
accuracy = accuracy_score(Y_test_gold, Y_test_predictions)
print(f"The performance of model in test dataset is {round(accuracy, 3)}")
