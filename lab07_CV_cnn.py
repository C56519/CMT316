import tensorflow as tf
import keras

# 1 加载数据集
fashion_mnist = keras.datasets.fashion_mnist
# 1.1 划分训练集、测试集
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
# 1.2 归一化
x_train = x_train / 255.0
x_test = x_test / 255.0

# 2 部署神经网络
# 2.1 早停法：损失在10个 epoch 内没有改善时，停止训练
early_stopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
# 2.2 配置神经网络各层
model = keras.models.Sequential([
    # 一、卷积层来提取特征
    # (1) 二维卷积层 Conv2D，参数：
    # 卷积核个数：64个卷积核，每个卷积核产生一个特征图
    # 卷积核大小：每个卷积核大小为 7*7 像素。每个卷积核会在输入图像上滑动，计算 7x7 区域内像素的加权和
    # 激活函数：RELU
    # padding="same"：在卷积操作后，输出图像的尺寸与输入图像相同。为了实现这一点，卷积核会在输入图像周围添加适当数量的零填充
    # 规定输入图像：大小为 28*28 像素，单通道(灰度图像)，如果为RGB图像，为 3
    keras.layers.Conv2D(64, 7, activation="relu", padding="same",
                        input_shape=[28, 28, 1]),
    # (2) 最大池化层
    # 作用：下采样（Downsampling，该技术是在处理图像或信号时，通过减少数据的分辨率或数量来简化数据。
    # 目的是既保留了重要的特征信息，又降低了数据的维度和计算复杂度。减小特征图的尺寸，从而减少计算量和模型复杂度。
    # 原理：创建一个2*2尺寸(一般2*2或3*3)的滑动窗口，在特征图上滑动，只取该区域内的最大值，形成新的特征图，者无疑将原来 28*28 像素的特征图变为了14*14
    keras.layers.MaxPooling2D(2),
    keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
    keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
    keras.layers.MaxPooling2D(2),
    keras.layers.Conv2D(256, 3, activation="relu", padding="same"),
    keras.layers.Conv2D(256, 3, activation="relu", padding="same"),
    keras.layers.MaxPooling2D(2),

    # 二、全连接层来完成分类任务
    # 将多维输入展平成一维。
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation="softmax")
])

# 3 训练和测试
# 3.1 配置损失函数、优化器、评估标准
# 损失函数为稀疏分类交叉熵损失
model.compile(loss = "sparse_categorical_crossentropy", optimizer="sgd", metrics = ["accuracy"])
# 3.2 训练
# 参数
# x_train.reshape((x_train.shape[0], 28, 28, 1)，规范训练数据，使其与模型配置的输入图像要求对接，
# x_train.shape[0]得到训练集图片数量，28*28像素，单通道。再使用x_train.reshape()变形
history = model.fit(x_train.reshape((x_train.shape[0], 28, 28, 1)), y_train, epochs = 60, validation_split=0.1, callbacks=[early_stopping_cb])
# 3.3 测试
# 同样要规范测试数据，使其与模型配置的输入图像要求对接
model.evaluate(x_test.reshape(x_test.shape[0], 28, 28, 1), y_test)