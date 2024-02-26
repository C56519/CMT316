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

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('omw-1.4')
nltk.download('wordnet')

# functions
def load_data():
    data_path = "../coursework/bbc"
    categories = [('business', 0), ('entertainment', 1), ('politics', 2), ('sport', 3), ('tech', 4)]
    news_list = []
    labels = []
    full_dataset = []
    for category in categories:
        category_path = os.path.join(data_path, category[0])
        for new in os.listdir(category_path):
            if new.endswith('.txt'):
                with open(os.path.join(category_path, new), encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    news_list.append(content)
                    labels.append(category[1])
                    full_dataset.append((content, category[1]))
    return news_list, labels, full_dataset

def get_word_list_in_sentence(string):
    lemmatizer = nltk.stem.WordNetLemmatizer()
    sentence_split = nltk.tokenize.sent_tokenize(string)
    word_list = []
    for sentence in sentence_split:
        pre_word_list = re.findall(r'\b\w+\b', sentence.lower())
        for word in pre_word_list:
            word_list.append(lemmatizer.lemmatize(word))
    return word_list

def create_dictionary(full_dataset, max_num_features):
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

def get_word_vector(vocabulary, string):
    word_vector = np.zeros(len(vocabulary))
    word_list = get_word_list_in_sentence(string)
    for index, word in enumerate(vocabulary):
        if word in word_list:
            word_vector[index] = word_list.count(word)  # Count the number of times the word appears in the sentence
    return word_vector

def log_reg_muti_training(full_dataset, dictionary):
    # Building the training set
    X_train = []
    Y_train = []
    for index, new in enumerate(full_dataset):
        new_vector = get_word_vector(dictionary, new[0])
        X_train.append(new_vector)
        Y_train.append(new[1])
    X_train = np.asarray(X_train)
    Y_train = np.asarray(Y_train)
    # training
    log_reg_muti = LogisticRegression(multi_class="multinomial", solver='lbfgs', C=10, random_state=42)
    log_reg_muti.fit(X_train, Y_train)
    return log_reg_muti
def kfold_training(training_and_dev_set, k):
    best_model = None
    highest_accuracy = 0
    best_dictionary = []
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
        dictionary = create_dictionary(training_set, 2000)
        # Training the model under this k
        kfold_log_reg_muti_model = log_reg_muti_training(training_set, dictionary)

        # (3) Use accuracy to validate the performance of the model under this k
        X_dev = []
        Y_dev = []
        for new in dev_set:
            word_vector = get_word_vector(dictionary, new[0])
            X_dev.append(word_vector)
            Y_dev.append(new[1])
        X_dev = np.asarray(X_dev)
        Y_dev_gold = np.asarray(Y_dev)
        Y_dev_predictions = kfold_log_reg_muti_model.predict(X_dev)
        accuracy = accuracy_score(Y_dev_gold, Y_dev_predictions)
        if accuracy > highest_accuracy:
            best_model = kfold_log_reg_muti_model
            highest_accuracy = accuracy
            best_dictionary = dictionary
        loop_count += 1
        print(f"The accuracy in {loop_count} training is: {round(accuracy, 3)}")
        print("======================================================================================")
        print(dictionary[:50])
        accuracy_all += accuracy
    # Find the average accuracy
    accuracy_average = round(accuracy_all / k, 3)
    print(f"The average accuracy in k-fold is: {round(accuracy_average, 3)}")
    print(f"Highest accuracy in k-fold is {round(accuracy_average, 3)}")
    return best_model, best_dictionary


# running
news_list, labels, full_dataset = load_data()
"""
for new in news_list[510:516]:
    print(new)
for category in labels[510:516]:
    print(category)
for category in full_dataset[900:906]:
    print(category)
"""
training_and_dev_set, test_set = train_test_split(full_dataset, test_size=0.2, random_state=42, shuffle=True)
clf_model, dictionary = kfold_training(training_and_dev_set, 5)

# test
X_test = []
Y_test = []
for new in test_set:
    word_vector = get_word_vector(dictionary, new[0])
    X_test.append(word_vector)
    Y_test.append(new[1])
X_test = np.asarray(X_test)
Y_test_gold = np.asarray(Y_test)
Y_test_predictions = clf_model.predict(X_test)
accuracy = accuracy_score(Y_test_gold, Y_test_predictions)
print(f"The performance of model in test dataset is {round(accuracy, 3)}")
