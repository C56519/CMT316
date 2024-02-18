[toc]

# Lab01-Lab03 Recap

## Lab01

第一个实验，初步引入了numpy和nltk，前者主要用于向量、矩阵、列表等数学计算。后者用于NLP自然语言处理

回顾用到的方法和函数：

````python
# 1 nltk
# 1.1 拆分字符串为指定结果
nltk.tokenize.word_tokenize("字符串")		用于将字符串拆分成单词列表
nltk.tokenize.sent_tokenize("字符串")		用于将字符串拆分成句子列表

# 1.2 词形还原：将单词还原为本身形式，如'cars' -> car
lemmatizer = nltk.stem.WordNetLemmatizer()
lemmatizer.lemmatize('单词字符串')


# 2 numpy
# 2.1 创建数组
array1 = np.zero(int) 创建指定大小的零列表
array2 = np.arange(start_index, finished_index, step_length) 创建等差列表
````



## Lab02

引入sklearn，用糖尿病的案例初步介绍机器学习项目。又以情感分析案例练习了特征提取(手动写，或用sklearn提供的卡方验证)，用来过滤训练集中的无效信息，只用有价值的信息进行训练，以减小噪声

### 糖尿病案例

```python
# 1 加载数据集
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
response = requests.get(url)
dataset_file = response.text.split("\n")

# 2 处理训练集，分为input和label两个列表
X_train = []
Y_train = []
for patient in dataset_file:
    this_patient_feature = []
    this_patient_label = []
    split_data = patient.split(",")
    for i in split_data[:-1]:
        this_patient_feature.append(float(i))
    this_patient_label.append(int(split_data[-1]))
    X_train.append(this_patient_feature)
    Y_train.append(this_patient_label)

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
```

### 情感分析案例

基本步骤与糖尿病案例一致，但是在加载完数据集后，添加了特征提取步骤，来减少训练集中的噪声项。

```python
# 1 特征提取
# 整体目标：创建全局词典，来选出1000个高频词，起到对样本进行过滤的作用
# 创建词典的步骤：
# (1) 有两个数据集，一个是pos，一个是neg，分别处理两个数据集
# (2) 对数据集中的每一条评论进行NLP处理：分词，剔除停用词来减少噪声
# (3) 统计剩下的单词的出现次数，合并到一个词典来存
# (4) 按照频率排序，选出频率最高的前1000个单词

# 1.1 分词函数
# 接收一段文本字符串，对其进行拆分成句子，每句拆分成单词，然后对每个单词进行词形还原，转小写。最终返回该段话的新列表
# [[句子1的第一个单词...句子1的最后一个单词] [句子2的第一个单词,...句子2的最后一个单词]...]
def get_word_list_in_sentence(string):
    lemmatizer = nltk.stem.WordNetLemmatizer()
    sentence_split = nltk.tokenize.sent_tokenize(string)
    word_list = []
    for sentence in sentence_split:
        pre_word_list = nltk.tokenize.word_tokenize(sentence)
        for word in pre_word_list:
            word_list.append(lemmatizer.lemmatize(word).lower())
    return word_list

# 1.2 创建词典
def create_dictionary(pos_dataset, neg_dataset, max_num_features):
    # 分词，剔除停用词，统计频率
    # (1) 规定停用词
    stopwords=set(nltk.corpus.stopwords.words('english'))
	stopwords.add('.')
	stopwords.add(',')
	stopwords.add("--")
	stopwords.add("``")
    word_count_list = {}
    for pos_review in pos_dataset:
        # (2) 分词
        word_list = get_word_list_in_sentence(pos_review)
        # (3) 剔除停用词
        for word in word_list:
            if word in stopwords:
                continue
            elif word in word_count_list:
                word_count_list[word] += 1
            else:
                word_count_list[word] = 1
    # (4) 统计单词出现频率
    for neg_review in neg_dataset:
        word_list = get_word_list_in_sentence(neg_review)
        for word in word_list:
            if word in stopwords:
                continue
            elif word in word_count_list:
                word_count_list[word] += 1
            else: word_count_list[word] = 1
    # 排序，创建词典
    sorted_list = sorted(word_count_list.items(), key=operator.itemgetter(1), reverse=True)[:max_num_features]
    dictionary = []
    for word, frequency in sorted_list:
        dictionary.append(word)
    return dictionary

# 2 开始处理训练集数据

# 2.1 将句子向量化
# 向量化：目的是将文本数据转换为机器学习算法能够理解和处理的数值形式。这里使用的是该单词在该句子中出现的频率
# 参数：上步创建的词典，需要转化成向量的句子
def get_word_vector(vocabulary, string):
    word_vector = np.zeros(len(vocabulary))
    word_list = get_word_list_in_sentence(string)
    for index, word in enumerate(vocabulary):
        if word in word_list:
            word_vector[index] = word_list.count(word)  # 计算该单词在句子中出现的次数
    return word_vector

# 3 训练
def svm_clf_training(pos_dataset, neg_dataset, dictionary):
    # 创建input和label列表
    X_train = []
    Y_train = []
    for pos_review in dataset_file_pos:
        # 将句子向量化
        pos_review_vector = get_word_vector(vocabulary, pos_review)
        X_train.append(pos_review_vector)
        Y_train.append(1)
    for neg_review in dataset_file_neg:
        neg_review_vector = get_word_vector(vocabulary, neg_review)
        X_train.append(neg_review_vector)
        Y_train.append(0)
    X_train = np.asarray(X_train)
    Y_train = np.asarray(Y_train)
    # 训练
    svm_clf = sklearn.svm.SVC(kernel="linear", gamma='auto')
    svm_clf.fit(X_train, Y_train)
    return svm_clf
```

当然，sklearn还提供了特征验证器，来直接对特征进行提取

```python
使用sklearn提供的卡方验证来高效的选择feature

from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest

# (1) 训练特征选择器
feature_selector = SelectKBest(chi2, k=500).fit(X_train, Y_train)

# (2) 使用训练好的特征选择器选择特征，实现过滤
X_train_new = feature_selector.transform(X_train)

# (3) 查看特征选择的结果
print(f"Size of original training matrix: {X_train.shape}")
print(f"Size of new traning matrix: {X_train_new.shape}")

# (4) 训练
svm_clf_new = sklearn.svm.SVC(kernel="linear", gamma='auto')
svm_clf_new.fit(X_train_new, Y_train)

# (6) 测试
sentence_3="Highly recommended: it was a fascinating film."
sentence_4="I got a bit bored, it was not what I was expecting."
print (svm_clf_new.predict(feature_selector.transform([get_word_vector(vocabulary,sentence_3)])))
print (svm_clf_new.predict(feature_selector.transform([get_word_vector(vocabulary,sentence_4)])))
```



## Lab03

综合了前两个实验的知识，新增了对训练集进行划分：训练集、验证集、测试集。所以较为完整的一个机器学习框架就建立起来了

1. 导入数据集
2. 划分数据集：训练集、验证集、测试集，并将其变为input、label 两个列表
3. 在训练集上训练模型
   - 创建全局词典
   - 向量化和训练
4. 在验证集上验证模型，参考几个值的结果，调整参数来调整模型
   - 参考的结果：精确率，召回率，f1分数，准确率，混淆矩阵，宏平均和微平均
5. 最后在测试集上测试模型：验证其方差与偏差，测试其性能(拟合度和泛化能力)
   - 参考的结果见上

### 情感分析案例

使用Lab02封装的几个函数：get_word_list_in_sentence, create_dictionary, get_word_vector, svm_clf_training

```python
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
```

验证和测试

```python
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
```

### 使用skearn进行k-fold交叉验证

```python
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
```







