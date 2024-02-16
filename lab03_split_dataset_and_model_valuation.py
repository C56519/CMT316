import numpy as np
import nltk
import sklearn
import operator
import requests
from sklearn.model_selection import train_test_split
import random
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score,recall_score,f1_score,accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('omw-1.4')
nltk.download('wordnet')


# 加载数据集
url_pos="http://josecamachocollados.com/rt-polarity.pos.txt" # Containing all positive reviews, one review per line
url_neg="http://josecamachocollados.com/rt-polarity.neg.txt" # Containing all negative reviews, one review per line
response_pos = requests.get(url_pos)
dataset_file_pos = response_pos.text.split("\n")
response_neg = requests.get(url_neg)
dataset_file_neg = response_neg.text.split("\n")

# 1 分割数据集：训练集，测试集，验证集
# (1) 建立整体数据集
full_dataset = []
for pos_review in dataset_file_pos:
    full_dataset.append((pos_review, 1))
for neg_review in dataset_file_neg:
    full_dataset.append((neg_review, 0))

# (2) 划分测试集和训练集
# 思想是要从整体数据集中，抽取10%数据放入测试集，其余的80%数据存入训练集中
# a: 随即生成一份列表，来确定哪些数据要放入测试集
# b: 遍历整体数据集，根据选出来的要放入测试集的索引列表，确定哪些元素放入测试集，哪些元素放入训练集
full_dataset_size = len(full_dataset)
test_dataset_size = int(round(full_dataset_size * 0.2, 0))   # round(数字， 保留几位小数)
# random.sample(list, number) 从列表list中随机抽样number个元素
# range(start_num, finished_num, footsteps): 生成指定范围和步长的数字列表
# 根据整体数据集的大小，使用range()生成一份从0开始到最后一个元素索引结束的索引列表，然后使用 random.sample() 随机选索引来确定哪些索引被指定放入测试集
test_data_index_list = random.sample(range(full_dataset_size), test_dataset_size)
training_set = []
test_set = []
for index, data in enumerate(full_dataset):
    if index in test_data_index_list:
        test_set.append(data)
    else: training_set.append(data)

# random.shuffle(list)  重拍列表元素原有的循序
random.shuffle(training_set)
random.shuffle(test_set)
print(f"Size of full dataset: {len(full_dataset)}")
print(f"Size of training dataset: {len(training_set)}")
print(f"Size of test dataset: {len(test_set)}")

'''
# 使用sklearn中的 train_test_split() 快速划分训练集、测试集
# 参数解释
# *arrays: 要被划分的数据集
# test_size: 要划分的测试集的比例
# train_size: 要划分数据集的比例。如果test_size写了，不写这个，默认按照互补比例算
# radom_state: 决定了数据集划分的随机性，默认None。随便设置一个int数字，这就保证了我们每次运行代码，都会得到相同的训练集、测试集划分结果
# shuffle: 是否打乱样本顺序，默认True

sk_traing_set, sk_test_set= train_test_split(full_dataset, test_size=0.2, random_state=0, shuffle=True)
print(len(sk_traing_set))
print(len(sk_test_set))
'''

# (3) 划分验证集
# 从测试集再划分50%作为验证集
validation_set_size = int(round(test_dataset_size * 0.5, 0))
validation_data_index_list = random.sample(range(test_dataset_size), validation_set_size)
new_test_set = []
validation_set = []
for index, data in enumerate(test_set):
    if index in validation_data_index_list:
        validation_set.append(data)
    else:
        new_test_set.append(data)
test_set = new_test_set

random.shuffle(training_set)
random.shuffle(test_set)
random.shuffle(validation_set)

print ("------------------------------")
print ("traning set:")
print ("Size of training set: "+str(len(training_set)))
for example in training_set[:3]:
  print (example)
print("------------------------------")
print ("validation set:")
print ("Size of validation set: "+str(len(validation_set)))
for example in validation_set[:3]:
  print (example)
print("------------------------------")
print ("test set:")
print ("Size of test set: "+str(len(test_set)))
for example in test_set[:3]:
  print (example)


# 2 使用lab2情感分析里的函数，在训练集、验证集、测试集上训练


# 2.1 训练集上训练
# 使用上个lab的函数
def get_word_list_in_sentence(string):
    lemmatizer = nltk.stem.WordNetLemmatizer()
    sentence_split = nltk.tokenize.sent_tokenize(string)
    word_list = []
    for sentence in sentence_split:
        pre_word_list = nltk.tokenize.word_tokenize(sentence)
        for word in pre_word_list:
            word_list.append(lemmatizer.lemmatize(word).lower())
    return word_list

stopwords=set(nltk.corpus.stopwords.words('english'))
stopwords.add('.')
stopwords.add(',')
stopwords.add("--")
stopwords.add("``")

def get_word_vector(vocabulary, string):
    word_vector = np.zeros(len(vocabulary))
    word_list = get_word_list_in_sentence(string)
    for index, word in enumerate(vocabulary):
        if word in word_list:
            word_vector[index] = word_list.count(word)  # 计算该单词在句子中出现的次数
    return word_vector

# 稍微修改下训练集
def create_dictionary(training_set, max_num_features):
    # 分词，剔除停用词，统计频率
    word_count_list = {}
    for review in training_set:
        word_list = get_word_list_in_sentence(review[0])
        for word in word_list:
            if word in stopwords:
                continue
            elif word in word_count_list:
                word_count_list[word] += 1
            else:
                word_count_list[word] = 1
    # 排序，创建词典
    sorted_list = sorted(word_count_list.items(), key=operator.itemgetter(1), reverse=True)[:max_num_features]
    dictionary = []
    for word, frequency in sorted_list:
        dictionary.append(word)
    return dictionary

def svm_clf_training(training_set, dictionary):
    # 构建训练集
    X_train = []
    Y_train = []
    for review in training_set:
        word_vector = get_word_vector(dictionary, review[0])
        X_train.append(word_vector)
        Y_train.append(review[1])
    X_train = np.asarray(X_train)
    Y_train = np.asarray(Y_train)
    # 训练
    svm_clf = sklearn.svm.SVC(kernel="linear", gamma='auto')
    svm_clf.fit(X_train, Y_train)
    return svm_clf

# 训练
distionary = create_dictionary(training_set, 1000)
first_svm_clf_training = svm_clf_training(training_set, distionary)

print(first_svm_clf_training.predict([get_word_vector(distionary, "It is good, fascinating!")]))


# 2.2 在测试集上测试
# 最后给出测试结果：精确率，召回率，f1分数，准确率，混淆矩阵，宏平均和微平均

# 整理测试集数据，返回两个列表，一个是input一个是真实label
def get_test_set_input_and_Y_test_gold(test_set, dictionary):
    X_test = []
    Y_test = []
    for review in test_set:
        word_vector = get_word_vector(distionary, review[0])
        X_test.append(word_vector)
        Y_test.append(review[1])
    X_test = np.asarray(X_test)
    Y_test_gold = np.asarray(Y_test)
    return X_test, Y_test_gold

# 开始测试
X_test, Y_test_gold = get_test_set_input_and_Y_test_gold(test_set, distionary)
Y_test_predictions = first_svm_clf_training.predict(X_test)
# 使用sklearn打印整体报告，该报告包含所有结果
print(classification_report(Y_test_gold, Y_test_predictions))
# 也可以单独计算
precision = precision_score(Y_test_gold, Y_test_predictions, average='macro')
recall = recall_score(Y_test_gold, Y_test_predictions, average='macro')
f1 = f1_score(Y_test_gold, Y_test_predictions, average='macro')
accuracy = accuracy_score(Y_test_gold, Y_test_predictions)
print(f"Precision: {round(precision, 3)}")
print(f"Recall: {round(recall, 3)}")
print(f"F1-Scoure: {round(f1, 3)}")
print(f"Accuracy: {round(accuracy, 3)}")
# 混淆矩阵
print (confusion_matrix(Y_test_gold, Y_test_predictions))


# 2.3 验证集上验证
# 为了模拟真实在机器学习项目过程，我们可以通过调整模型参数，来使得模型的偏差和方差维持在一个适当的情况。本实验中，我们可以以调整全局词典的高频关键字的个数，结合验证集来选择一个最佳模型

# (1) 建立验证集真实label列表
Y_validation = []
for review in validation_set:
    Y_validation.append(review[1])
Y_validation_gold = np.asarray(Y_validation)

# (2) 调整全局词典的高频关键字的个数，结合验证集来选择一个最佳模型
dictionary_max_feature = [250, 500, 750, 1000]
best_accurary_in_validation = 0
for max_feature_num in dictionary_max_feature:
    # 先在训练集上训练模型
    dictionary = create_dictionary(training_set, max_feature_num)
    new_svm_slf = svm_clf_training(training_set, dictionary)
    # 建立验证集input列表
    X_validation = []
    for review in validation_set:
        word_vector = get_word_vector(dictionary, review[0])
        X_validation.append(word_vector)
    X_validation = np.asarray(X_validation)
    # 让模型在验证集上跑，得到预测结果
    Y_validation_predictions = new_svm_slf.predict(X_validation)
    # 测评准确率
    validation_accuracy = accuracy_score(Y_validation_gold, Y_validation_predictions)
    print(f"Accuracy with {max_feature_num} features in dictionary: {round(validation_accuracy, 3)}")
    # 找出最佳模型对应的全局词典的高频关键字的个数
    if validation_accuracy >= best_accurary_in_validation:
        best_accurary_in_validation = validation_accuracy
        best_feature_num_in_dictionary = max_feature_num
        best_dictionary = dictionary
        best_svm_clf = new_svm_slf
print(f"Best accuracy overall in the validation set is: {round(best_accurary_in_validation, 3)}, with {best_feature_num_in_dictionary} features,")

# (3) 使用最佳参数，跑一次测试集，查看最终的训练结果
X_test, Y_test_gold = get_test_set_input_and_Y_test_gold(test_set, best_dictionary)
Y_test_predictions = best_svm_clf.predict(X_test)
print(classification_report(Y_test_gold, Y_test_predictions))



# 3 练习：同样是调成上述参数，这次列表为[100, 500, 1000]，而且这次不使用准确率来选最佳模型，换成参考f1-score来选，并尝试封装成函数
def validation_and_test_with_f1(training_set, validation_set, test_set, parameter_list):
    Y_validation = []
    for review in validation_set:
        Y_validation.append(review[1])
    Y_validation_gold = np.asarray(Y_validation)

    best_f1_in_validation = 0
    for max_feature_num in parameter_list:
        dictionary = create_dictionary(training_set, max_feature_num)
        new_svm_slf = svm_clf_training(training_set, dictionary)

        X_validation = []
        for review in validation_set:
            word_vector = get_word_vector(dictionary, review[0])
            X_validation.append(word_vector)
        X_validation = np.asarray(X_validation)

        Y_validation_predictions = new_svm_slf.predict(X_validation)
        # 测评f1
        validation_f1 = f1_score(Y_validation_gold, Y_validation_predictions)
        print(f"F1-Score with {max_feature_num} features in dictionary: {round(validation_f1, 3)}")

        if validation_f1 >= best_f1_in_validation:
            best_f1_in_validation = validation_f1
            best_feature_num_in_dictionary = max_feature_num
            best_dictionary = dictionary
            best_svm_clf = new_svm_slf
    print(
        f"Best f1 overall in the validation set is: {round(best_f1_in_validation, 3)}, with {best_feature_num_in_dictionary} features,")

    X_test, Y_test_gold = get_test_set_input_and_Y_test_gold(test_set, best_dictionary)
    Y_test_predictions = best_svm_clf.predict(X_test)
    print(classification_report(Y_test_gold, Y_test_predictions))


parameter_list = [100, 500, 1000]
validation_and_test_with_f1(training_set, validation_set, test_set, parameter_list)





# 4 使用 sklearn 中的k-fold来进行交叉验证，快速验证
def kfold_training(full_dataset, k):
    # 创建k-fold交叉验证器，并设置 k
    kfold = KFold(n_splits=k)
    # 打散原始数据，保证数据的随机性
    random.shuffle(full_dataset)
    # 直接生成所有 k-fold 情况下训练集、测试集数据的索引列表
    all_k_index_list_of_training_set_and_test_set = kfold.split(full_dataset)
    # 迭代每个k下，进行训练和验证
    accuracy_all = 0
    loop_count = 0
    for this_k_training_set_index_list, this_k_test_set_index_list in all_k_index_list_of_training_set_and_test_set:
        training_set = []
        test_set = []
        # (1) 创建本次k下的训练集和测试集列表
        for index, review in enumerate(full_dataset):
            if index in this_k_test_set_index_list:
                training_set.append(review)
            else:
                test_set.append(review)
        # (2) 训练本次k下的模型
        # 创建本次k下的全局保留关键词的词典
        dictionary = create_dictionary(training_set, 500)
        # 训练本次k下的模型
        kfold_svm_clf = svm_clf_training(training_set, dictionary)

        # (3) 使用准确率测试本次k下的模型性能
        X_test = []
        Y_test = []
        for review in test_set:
            word_vector = get_word_vector(dictionary, review[0])
            X_test.append(word_vector)
            Y_test.append(review[1])
        X_test = np.asarray(X_test)
        Y_test_gold = np.asarray(Y_test)
        Y_test_predictions = kfold_svm_clf.predict(X_test)
        accuracy = accuracy_score(Y_test_gold, Y_test_predictions)
        loop_count += 1
        print(f"The accuracy in {loop_count} training is: {round(accuracy, 3)}")
        accuracy_all += accuracy
    # 求平均准确率
    accuracy_average = round(accuracy_all / k, 3)
    print(f"The average accuracy in k-fold is: {round(accuracy_average, 3)}")

kfold_training(full_dataset, 5)







