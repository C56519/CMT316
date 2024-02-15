import numpy as np
import nltk
import sklearn
import operator
import requests
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')


# 数据集需要自己提取特征的情况
# 本例中，使用数据集进行情感分析。将用户的评论分类为正面或负面


# 1 加载数据集
url_pos="http://josecamachocollados.com/rt-polarity.pos.txt"
url_neg="http://josecamachocollados.com/rt-polarity.neg.txt"
# 正面评论数据集
response_pos = requests.get(url_pos)
dataset_file_pos = response_pos.text.split("\n")
# 负面评论数据集
response_neg = requests.get(url_neg)
dataset_file_neg = response_neg.text.split("\n")
# 查看前五个数据
print("postive reviews:")
for index, pos_review in enumerate(dataset_file_pos[:5]):
    print(f"user{index}: {pos_review}")
print("negitive reviews:")
for index, neg_review in enumerate(dataset_file_neg[:5]):
    print(f"user{index}: {neg_review}")


# 2 对数据集进行预处理
# 整体目标：创建全局词典，来选出1000个高频词，起到对样本进行过滤的作用
# 创建词典的步骤：
# (1) 有两个数据集，一个是pos，一个是neg，分别处理两个数据集
# (2) 对数据集中的每一条评论进行NLP处理：分词，剔除停用词来减少噪声
# (3) 统计剩下的单词的出现次数，合并到一个词典来存
# (4) 按照频率排序，选出频率最高的前1000个单词

# 2.1 使用nltk进行分词
# 先将一段话拆成句子，再将每句话分割成单词列表, 然后进行词干提取并转成小写
def get_word_list_in_sentence(string):
    lemmatizer = nltk.stem.WordNetLemmatizer()
    sentence_split = nltk.tokenize.sent_tokenize(string)
    word_list = []
    for sentence in sentence_split:
        pre_word_list = nltk.tokenize.word_tokenize(sentence)
        for word in pre_word_list:
            word_list.append(lemmatizer.lemmatize(word).lower())
    return word_list

# 2.2 剔除停用词
# nltk.corpus.stopwords.words('english') 剔除停用词
# 停用词是英语中不重要的词汇，比如the”、“is”、“in”等。在NLP中，移除停用词是一个常见的预处理步骤，可以帮助减少数据的噪声，使模型更加关注于那些有意义的词汇。
# 这里使用 set() 将list转换成集合(set)，原因是集合在进行查找操作（例如，判断某个词是否为停用词）时比列表更加高效
stopwords=set(nltk.corpus.stopwords.words('english'))
stopwords.add('.')
stopwords.add(',')
stopwords.add("--")
stopwords.add("``")

# 2.3 统计各个词汇出现的频率
word_count_list = {}
for pos_review in dataset_file_pos:
    word_list = get_word_list_in_sentence(pos_review)
    for word in word_list:
        if word in stopwords:
            continue
        elif word in word_count_list:
            word_count_list[word] += 1
        else:
            word_count_list[word] = 1
for neg_review in dataset_file_neg:
    word_list = get_word_list_in_sentence(neg_review)
    for word in word_list:
        if word in stopwords:
            continue
        elif word in word_count_list:
            word_count_list[word] += 1
        else:
            word_count_list[word] = 1

# 2.4 按照频率排序，并挑出1000个高频词
sorted_list = sorted(word_count_list.items(), key=operator.itemgetter(1), reverse=True)[:1000]

# 2.5 创建词典
vocabulary = []
for word, frequency in sorted_list:
    vocabulary.append(word)


# 3 将句子向量化
# (1) 对照全局词典，剔除不重要的单词
# (2) 向量化：将该该句子中剩下的单词对应一个数字，目的是将文本数据转换为机器学习算法能够理解和处理的数值形式。这里使用的是该单词在该句子中出现的频率
def get_word_vector(vocabulary, string):
    word_vector = np.zeros(len(vocabulary))
    word_list = get_word_list_in_sentence(string)
    for index, word in enumerate(vocabulary):
        if word in word_list:
            word_vector[index] = word_list.count(word)  # 计算该单词在句子中出现的次数
    return word_vector


# 4 开始构建最终要用于训练的数据集
X_train = []
Y_train = []
for pos_review in dataset_file_pos:
    pos_review_vector = get_word_vector(vocabulary, pos_review)
    X_train.append(pos_review_vector)
    Y_train.append(1)
for neg_review in dataset_file_neg:
    neg_review_vector = get_word_vector(vocabulary, neg_review)
    X_train.append(neg_review_vector)
    Y_train.append(0)

X_train = np.asarray(X_train)
Y_train = np.asarray(Y_train)


# 5 训练模型
svm_clf = sklearn.svm.SVC(kernel="linear", gamma='auto')
svm_clf.fit(X_train, Y_train)


# 6 测试
sentence_1="Fascinating, I loved it."
sentence_2="Bad movie, probably one of the worst I have ever seen."
print (svm_clf.predict([get_word_vector(vocabulary,sentence_1)]))
print (svm_clf.predict([get_word_vector(vocabulary,sentence_2)]))



# 7 练习：封装两个函数，一个用来过滤(三个参数：pos训练集，neg训练集，选择高频词的数量)，一个用来训练。来实现上述步骤。
def create_dictionary(pos_dataset, neg_dataset, max_num_features):
    # 分词，剔除停用词，统计频率
    word_count_list = {}
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
    # 排序，创建词典
    sorted_list = sorted(word_count_list.items(), key=operator.itemgetter(1), reverse=True)[:max_num_features]
    dictionary = []
    for word, frequency in sorted_list:
        dictionary.append(word)
    return dictionary

def svm_clf_training(pos_dataset, neg_dataset, dictionary):
    # 构建训练集
    X_train = []
    Y_train = []
    for pos_review in dataset_file_pos:
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

# 测试
ex_dictionary = create_dictionary(dataset_file_pos, dataset_file_neg, 2000)
svm_clf = svm_clf_training(dataset_file_pos, dataset_file_pos, ex_dictionary)
print("Exercise results:")
print (svm_clf.predict([get_word_vector(vocabulary,sentence_1)]))
print (svm_clf.predict([get_word_vector(vocabulary,sentence_2)]))






# 8 另一种方法：使用sklearn提供的卡方验证来高效的选择feature

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


