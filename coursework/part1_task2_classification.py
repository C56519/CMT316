import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score

# 1 load the dataset
training_set_path = '../coursework/real-state/train_full_Real-estate.csv'
test_set_path = '../coursework/real-state/test_full_Real-estate.csv'
training_dataset_file = open(training_set_path).readlines()
test_dataset_file = open(test_set_path).readlines()

# 2 define two functions
# 2.1 deal with the input and label
def get_input_and_label(data_set):
    X_list = []
    Y_list = []
    for house in data_set[1:]:
        this_house_features = []
        this_house_label = []
        split_data = house.split(',')
        for i in split_data[:-1]:
            this_house_features.append(float(i))
        if float(split_data[-1].strip()) >= 30:
            this_house_label.append(1)
        else: this_house_label.append(0)
        X_list.append(this_house_features)
        Y_list.append(this_house_label)
    X_list = np.asarray(X_list)
    Y_list = np.asarray(Y_list)
    return X_list, Y_list

# 2.2 using grid search and pipeline to select the best model
"""
results:
The best parameter: {'svm_clf__C': 2.9499999999999997}
The best accuracy in training set: 0.9381505376344087
"""
def searching_best_clf_model(X_train, Y_train):
    parameters_space = {
        "svm_clf__C": np.arange(0.01, 5, 0.01),
        "svm_clf__tol": np.arange(0.01, 0.11, 0.01),
    }
    pre_clf_model = Pipeline([
        ("scalar", StandardScaler()),
        ("svm_clf", SVC(kernel="rbf", gamma="scale", C=2, tol=1e-3))
    ])
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
    search = GridSearchCV(pre_clf_model, parameters_space, scoring='accuracy', cv=cv, n_jobs=-1)
    best_model = search.fit(X_train, Y_train)
    print("Grid Search:")
    print(f"The best parameter: {search.best_params_}")
    print(f"The best accuracy in training set: {search.best_score_}")
    return best_model

# 3 training
X_train, Y_train = get_input_and_label(training_dataset_file)
clf_model = searching_best_clf_model(X_train, Y_train.ravel())

# 4 test
X_test, Y_test_gold = get_input_and_label(test_dataset_file)
Y_test_prediction = clf_model.predict(X_test)
accuracy = accuracy_score(Y_test_gold, Y_test_prediction)
print("\nPerformance of the model on the test set:")
print(f"Accuracy: {accuracy}")