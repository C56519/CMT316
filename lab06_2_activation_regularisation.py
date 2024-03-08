import tensorflow as tf
import keras
import matplotlib.pyplot as plt


# 1 比较使用不同激活函数后的影响


# 1.1 加载数据并处理
fashion_mnist = keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
x_train = x_train / 255.0
x_test = x_test / 255.0

# 1.2 训练三个神经网络，使用不同激活函数
# ReLU
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax"))
model.compile(loss = "sparse_categorical_crossentropy", optimizer="sgd", metrics = ["accuracy"])
history_ReLU = model.fit(x_train, y_train, epochs = 30, validation_split=0.1)
model.evaluate(x_test, y_test)

# Tanh
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
model.add(keras.layers.Dense(300, activation="tanh"))
model.add(keras.layers.Dense(100, activation="tanh"))
model.add(keras.layers.Dense(10, activation="softmax"))
model.compile(loss = "sparse_categorical_crossentropy", optimizer="sgd", metrics = ["accuracy"])
history_Tanh = model.fit(x_train, y_train, epochs = 30, validation_split=0.1)
model.evaluate(x_test, y_test)

# LeakyReLU
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
model.add(keras.layers.Dense(300, activation=keras.layers.LeakyReLU()))
model.add(keras.layers.Dense(100, activation=keras.layers.LeakyReLU()))
model.add(keras.layers.Dense(10, activation="softmax"))
model.compile(loss = "sparse_categorical_crossentropy", optimizer="sgd", metrics = ["accuracy"])
history_LeakyReLU = model.fit(x_train, y_train, epochs = 30, validation_split=0.1)
model.evaluate(x_test, y_test)

# 1.3 画图
# 准确率曲线
plt.plot(history_Tanh.history['accuracy'])
plt.plot(history_ReLU.history['accuracy'])
plt.plot(history_LeakyReLU.history['accuracy'])
plt.title('Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['tanh', 'ReLU', 'Leaky ReLU'], loc='upper left')
plt.show()

# Loss曲线
plt.plot(history_Tanh.history['loss'])
plt.plot(history_ReLU.history['loss'])
plt.plot(history_LeakyReLU.history['loss'])
plt.title('Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['tanh', 'ReLU', 'Leaky ReLU'], loc='upper left')
plt.show()


# 2 早期停止 和 Dropout 技术

# 2.1 早期停止
# 神经网络的一种正则化手段，以防止过拟合，会在验证集的loss或者准确率等规定好的指标，不再改善时停止训练来提前结束模型的训练
# keras.callbacks.EarlyStopping() 创建一个早期停止的回调函数对象
# 参数
# patience=10: 如果在连续 10 个 epoch 中验证集的损失没有降低，则停止训练
# restore_best_weights=True，表示在停止训练后选出验证集上性能最佳的模型参数
early_stopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
# 2.2 Dropout
# 神经网络的另一个正则化手段，以防止过拟合，会在训练过程中随机停用一定比率的神经元
# 使用原因
# 1. 减少依赖：神经网络模型有时候会变得对训练数据中的特定模式过于敏感，从而导致模型泛化能力弱。通过Dropout，使之学会不依赖于任何一组特征。
# 2. 相当于集成训练：因为Dropout的随机性，所以相当于训练了多个不同的神经网络。
# keras.layers.Dropout() 参数
# rate=0.2 代表每个训练步骤中，每个神经元有0.2的概率被临时停用
model.add(keras.layers.Dropout(rate=0.2))
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dropout(rate=0.2))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dropout(rate=0.2))
model.add(keras.layers.Dense(10, activation="softmax"))
model.compile(loss = "sparse_categorical_crossentropy", optimizer="sgd", metrics = ["accuracy"])
# 早期停止：callbacks=[early_stopping_cb] 将早期停止的回调函数添加到训练过程中。
ReLU_history = model.fit(x_train, y_train, epochs = 60, validation_split=0.1, callbacks=[early_stopping_cb])
model.evaluate(x_test, y_test)