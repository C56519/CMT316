import tensorflow as tf
import keras
import matplotlib.pyplot as plt

# 1 加载数据并处理
fashion_mnist = keras.datasets.fashion_mnist
# 划分训练集和测试集
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
# 将数据从 (0. 255) 映射到 (0, 1)
x_train = x_train / 255.0
x_test = x_test / 255.0

# 2 定义神经网络
# 2.1 创建keras神经网络顺序模型
model = keras.models.Sequential()
# 2.2 规定各层
# 告诉flatten层将28 * 28的输入数据转成1维
model.add(keras.layers.Flatten(input_shape=[28, 28]))
# 隐藏层有三层
# 对于每一层
# 1. 隐藏层1：128 个神经元，激活函数ReLU。这一层将接收来自上一层的展平后的输入数据，并输出一个长度为 128 的向量。
# 2. 加入批量归一化层：在激活函数之前应用，对每个批次的输入进行归一化处理，使得输入的均值接近0，标准差接近1。
# 这有助于加速神经网络的训练过程，减少梯度消失问题，并使模型对超参数的选择更加稳定。
# 3. 使用L2正则化 kernel_regularizer，避免过拟合
model.add(keras.layers.BatchNormalization()) # 批量归一化层
model.add(keras.layers.Dense(300, activation="relu", kernel_regularizer=keras.regularizers.l2(0.01))) 

model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(100, activation="relu", kernel_regularizer=keras.regularizers.l2(0.01)))

model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(10, activation="softmax"))

# 2.3 查看
# 打印出模型的信息，包括模型的结构、每一层的输出形状和参数数量等
model.summary()

# 3 训练
# 3.1 定义训练器

# model.compile() 参数如下
# loss: 损失函数。这里使用的是稀疏分类交叉熵损失函数，适用于多类别分类问题，并且标签是整数编码的情况。
# optimizer: 模型函数，随机梯度下降（SGD）优化器来最小化 loss, 设置了learning_rate学习率
# metrics: 衡量标准，这里使用 accuracy
model.compile(loss = "sparse_categorical_crossentropy", optimizer=keras.optimizers.SGD(learning_rate=0.01), metrics = ["accuracy"])
# 另一个模型函数 
# model.compile(loss = "sparse_categorical_crossentropy", optimizer=keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999), metrics = ["accuracy"])
# 3.2 训练
# model.fit() 参数如下
# x_train: 通常是一个NumPy数组或者一个TensorFlow的张量（Tensor）
# epochs = 30: 这指定了训练过程中全数据集的遍历次数。一个epoch意味着每个样本在更新模型权重前都被预测了一次。
# validation_split: 训练过程中划分验证集 10%
muti_clf = model.fit(x_train, y_train, epochs = 30, validation_split=0.1)

# 4 测试
model.evaluate(x_test, y_test)



# 4 画图
# 4.1 图一：训练集和验证集上的准确率变化曲线
plt.plot(muti_clf.history['accuracy'])
plt.plot(muti_clf.history['val_accuracy'])
# 标题，x y 轴的label
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
# 图例
# plt.lengend() 参数如下
# ['Train', 'Val'] 图例上显示两个标签，分别表示训练集和验证集
# loc='upper left' 指定了图例的位置，这里是在图的左上角。
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()

# 4.2 图二：训练集和验证集上的损失变化曲线
plt.plot(muti_clf.history['loss'])
plt.plot(muti_clf.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()
