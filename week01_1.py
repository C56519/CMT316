import numpy as np
import nltk

# 下载依赖包
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

# 1 sltk
# 1.1 定义一些段落
sentence1 = "Machine learning is the scientific study of algorithms and statistical models that computer systems use to perform a specific task without using explicit instructions, relying on patterns and inference instead. "
sentence2 = "It is seen as a subset of artificial intelligence. "
sentence3 = "Machine learning algorithms build a mathematical model based on sample data, known as training data, in order to make predictions or decisions without being explicitly programmed to perform the task. "
paragraph = sentence1 + sentence2 + sentence3
print(paragraph)


# 1.2 nltk.tokenize.word_tokenize() 用于将文本字符串拆分成单词列表
"""
函数背后使用了 PunktSentenceTokenizer，这是一个无监督的可训练模型，用于将文本拆分成单词列表。它能够处理复杂的单词切分情况，比如缩写、带点的名字等。该函数返回一个单词列表，包括标点符号。
1. 分词结果包括标点符号。如果你想要纯粹的单词列表，可能需要进一步处理这些标点符号。
2. 默认对英文文本效果最佳。对于其他语言，可能需要使用特定于语言的分词器。
3. 在处理大量文本数据时，分词可能会成为性能瓶颈。
"""
list_tokens=nltk.tokenize.word_tokenize(paragraph)
print(list_tokens)


# 1.3 nltk.tokenize.sent_tokenize() 将文本拆分成句子列表
sentence_split = nltk.tokenize.sent_tokenize(paragraph)
print(sentence_split)


# 1.4 练习：三个句子中，输出含有 learning 的句子数量
# (1) 先将各个句子拆成自己的单词列表
list_sentence_tokens = []
for sentence in sentence_split:
    list_sentence_tokens.append(nltk.tokenize.word_tokenize(sentence))
# (2) 检测
count = 0
for sentence in list_sentence_tokens:
    if 'learning' in sentence:
        count += 1
print(f"Number of sentences containing 'learning' is {count}")


# 1.5 nltk.stem.WordNetLemmatizer()
"""
词形还原(Lemmatization)与词干提取(stemming)
词形还原是把单词还原成本身的形式：比如将‘cars’还原成car，把‘ate’还原成‘eat’，把‘handling’还原成‘handle’
词干提取则是提取单词的词干，比如将‘cars’提取出‘car’，将‘handling’提取出来‘handl’（单纯的去掉ing），对于‘ate’使用词干提取则不会有任何的效果。
"""
# 使用：先初始化，用的时候.lemmatize(string)
lemmatizer = nltk.stem.WordNetLemmatizer()
lower_case = []
for sentence in list_sentence_tokens:
    sentence_temp_list = []
    for word in sentence:
        sentence_temp_list.append(lemmatizer.lemmatize(word).lower())
    lower_case.append(sentence_temp_list)
print(lower_case)

# 1.6 练习：封装一个函数，来实现对一段话 小写每个单词
def get_list_word(paragraph):
    sentence_split = nltk.tokenize.sent_tokenize(paragraph)
    word_list = []
    for sentence in sentence_split:
        word_list_temp = nltk.tokenize.word_tokenize(sentence)
        for word in word_list_temp:
            word_list.append(lemmatizer.lemmatize(word).lower())
    return word_list



# 2 numpy
# 2.1 创建数组
# .zeros(num) 创建指定大小的零数组
# .arrarge(start_index, finish_index, step_length) 创建指定等差的列表
# 默认从0开始到终止索引结束(不包括)。如果只有一个参数，参数默为终止索引。默认步长为 1
array1 = np.zeros(3)
array2 = np.arange(3)
print(array1)
print(array2)

# 2.2 结合 nltk 创建两个数字列表，一个记录所有去过重的单词，一个记录他们对应的出现次数
# 空字典来存储所有去过重的单词和出现次数
count_dic = {}
# 如果字典中出现该单词，次数加1；如果没出现，添加到字典，次数设为一
for sentence in lower_case:
    for word in sentence:
        if word in count_dic: count_dic[word] += 1
        else: count_dic[word] = 1
print(count_dic)
# 创建列表来存储出现次数
word_count_list = np.zeros(len(count_dic))
# 创建列表来存储去过重的单词
new_word_list = list(count_dic.keys())
print(new_word_list)
for i in range(len(new_word_list)):
    word_count_list[i] = count_dic[new_word_list[i]]
print(word_count_list)



