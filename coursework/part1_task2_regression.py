import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RepeatedKFold
from sklearn.metrics import mean_squared_error

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
        this_house_label.append(float(split_data[-1].strip()))
        X_list.append(this_house_features)
        Y_list.append(this_house_label)
    X_list = np.asarray(X_list)
    Y_list = np.asarray(Y_list)
    return X_list, Y_list

# 2.2 using grid search and pipeline to select the best model
"""
results:
The best parameter: {'poly_features__degree': 2, 'regul_reg__alpha': 0.99}
The best RMSE in training set: 7.763544251303016
"""
def searching_best_model(X_train, Y_train):
    parameters_space = {
        "poly_features__degree": np.arange(1, 5),
        "regul_reg__alpha": np.arange(0, 1, 0.01)
    }
    pre_model = Pipeline([
        ("poly_features", PolynomialFeatures(include_bias=False)),
        ("std_scaler", StandardScaler()),
        ("regul_reg", Ridge(solver="svd", alpha=1, fit_intercept=True, random_state=42))
    ])
    cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=1)
    search = GridSearchCV(pre_model, parameters_space, scoring='neg_root_mean_squared_error', cv=cv, n_jobs=-1)
    best_model = search.fit(X_train, Y_train)
    print(f"The best parameter: {search.best_params_}")
    print(f"The best RMSE in training set: {-search.best_score_}")
    return best_model

# 3 training
X_train, Y_train = get_input_and_label(training_dataset_file)
reg_model = searching_best_model(X_train, Y_train)

# 4 test
X_test, Y_test_gold = get_input_and_label(test_dataset_file)
Y_test_prediction = reg_model.predict(X_test)
mse = mean_squared_error(Y_test_gold, Y_test_prediction)
rmse = np.sqrt(mse)
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")