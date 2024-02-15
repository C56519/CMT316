import numpy as np
import nltk
import sklearn
import operator
import requests
from sklearn.model_selection import train_test_split
import random
nltk.download('stopwords') # If needed
nltk.download('punkt') # If needed
nltk.download('omw-1.4') # If needed
nltk.download('wordnet') # If needed


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
