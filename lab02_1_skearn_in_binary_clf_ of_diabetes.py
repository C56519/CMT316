import numpy as np
import nltk
import sklearn
import requests
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')


# 本实验，利用sklearn来进行第一次机器学习训练
# 我们将使用一个二元分类数据集，有8个特征，1个标签。共有768个病人的数据。目标是预测一个人是否患有糖尿病


# 1 加载数据集
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
response = requests.get(url)
dataset_file = response.text.split("\n")

# 1.1 查看数据集
print(f"Number of patients: {len(dataset_file)}")
print("The information of the first five patients:")
for index, people in enumerate(dataset_file[:5]):
    print(f"parent{index + 1}: {people}")


# 2 数据预处理
X_train = []
Y_train = []
for parent in dataset_file:
    this_parent_feature = []
    this_parent_label = []
    split_data = parent.split(",")
    for i in split_data[:-1]:
        this_parent_feature.append(float(i))
    this_parent_label.append(int(split_data[-1]))
    X_train.append(this_parent_feature)
    Y_train.append(this_parent_label)

X_train = np.asarray(X_train)
Y_train = np.asarray(Y_train)


# 3 训练
svm_clf = sklearn.svm.SVC(kernel="linear", gamma='auto')
svm_clf.fit(X_train, Y_train)


# 4 测试
patient_1=['0', '100', '86', '20', '39', '35.1', '0.242', '21']
patient_2=['1', '197', '70', '45', '543', '30.5', '0.158', '51']
print (svm_clf.predict([patient_1]))
print (svm_clf.predict([patient_2]))



# 5 练习
# 从所有特征中选择三个特征，来进行训练
ex_x_train = []
ex_y_train = []
selected_features = [0, 3, 5]
for parent in dataset_file:
    this_parent_feature = []
    this_parent_label = []
    split_data = parent.split(",")
    for i in selected_features:
        this_parent_feature.append(float(split_data[i]))
    ex_x_train.append(this_parent_feature)
    ex_y_train.append(int(split_data[-1]))

ex_x_train = np.asarray(ex_x_train)
ex_y_train = np.asarray(ex_y_train)

ex_svm_clf = sklearn.svm.SVC(kernel="linear", gamma='auto')
ex_svm_clf.fit(ex_x_train, ex_y_train)

patient_1=['3', '35.2', '51']
patient_2=['1', '20.5', '21']
print (ex_svm_clf.predict([patient_1]))
print (ex_svm_clf.predict([patient_2]))
