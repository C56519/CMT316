import tensorflow as tf
import keras
from google.colab import drive
import keras.utils as image
import tensorflow_datasets as tfds

# 使用预训练网络进行训练
# 预训练网络：已经在某个大型数据集上训练好的神经网络模型

# 1 使用 resent50 预训练网络
model = keras.applications.resnet50.ResNet50(weights="imagenet")
# 链接谷歌驱动
drive.mount('/content/gdrive')
# 打印模型概况
model.summary()

# 2 测试模型准确率
# 2.1 加载一张图片，并调整大小, 将图像调整为224x224像素，这通常是许多预训练模型的输入尺寸
img = image.load_img("/content/gdrive/My Drive/IndianElephant.jpg", target_size=(224, 224))
# 将图片转换为 numpy 数组
img = image.img_to_array(img)
# 重新调整图像数据的形状
# 这里 (1, 224, 224, 3) 表示 1 张 224x224 大小的 RGB 图像。
img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
# 对图像数据进行预处理，使其符合ResNet50模型的输入要求。这通常包括归一化像素值等操作
img = keras.applications.resnet50.preprocess_input(img)
# 2.2 开始预测图片分类，输出每个类别的概率 
Y_prob = model.predict(img)
# 2.3 解码并显示结果
# 将模型的预测结果转换为可读的类别名称。top=3 表示取前3个预测结果。
top_K = keras.applications.resnet50.decode_predictions(Y_prob, top=3)
# 遍历预测结果并输出类别 ID、名称和概率
for class_id, name, y_proba in top_K[0]:
  print("class_id:", class_id, "name:", name, " ", y_proba*100, "%")

# 3 微调预训练模型
# 3.1 准备数据集
# 加载数据集
dataset, info = tfds.load("tf_flowers", as_supervised=True, with_info=True)
# 打印数据集大小
dataset_size = info.splits["train"].num_examples
print("size: ", dataset_size)
# 打印所有类别名称
class_names = info.features["label"].names
print("classes: ", class_names)
# 打印类别的数量
n_classes = info.features["label"].num_classes
print("num. classes: ", n_classes)

# 3.2 划分训练集、验证集、测试集
# as_supervised=True：将数据集加载为 (image, label) 对，即 (图像, 标签)
test_set = tfds.load("tf_flowers", split="train[:10%]", as_supervised=True)     # 前10%测试集
valid_set = tfds.load("tf_flowers", split="train[10%:25%]", as_supervised=True)     # 10%-25%验证集
train_set = tfds.load("tf_flowers", split="train[25%:]", as_supervised=True)    # 剩余作为

# 3.3 数据预处理
def preprocess(image, label):
  # 调整图像尺寸224*224,以满足预处理模型的输入图像尺寸
  resized_image = tf.image.resize(image, [224, 224])
  # 使用 Xception 模型的预处理函数对图像进行预处理，通常包括像素值归一化等操作
  final_image = keras.applications.xception.preprocess_input(resized_image)
  return final_image, label

# 3.4 其他配置
# 设置批处理大小
batch_size = 32
# 将训练数据随机打乱
train_set = train_set.shuffle(1000)
# 对训练数据应用预处理函数
# .map(preprocess) 对训练数据应用预处理函数
# ..repeat 重复数据集，以便在多个 epoch 中使用
# .batch(batch_size)：将数据集分割成批次，每批包含 batch_size 个样本
# .prefetch(1)：预取 1个批次的数据，以提高训练效率
train_set = train_set.map(preprocess).repeat().batch(batch_size).prefetch(1)
valid_set = valid_set.map(preprocess).repeat().batch(batch_size).prefetch(1)
test_set = test_set.map(preprocess).batch(batch_size).prefetch(1)

# 3.5 部署预训练模型
# 使用Xception模型
# weights="imagenet"：加载在ImageNet数据集上预训练的权重
# include_top=False：移除顶层的 全连接层，这允许我们添加自定义的全连接层以适应新任务
base_model = keras.applications.xception.Xception(weights="imagenet", include_top=False)
# 添加全局平均池化层
avg = keras.layers.GlobalAveragePooling2D()(base_model.output)
# 添加自定义层
# keras.layers.Dense：添加一个全连接层
# n_classes：输出层的神经元数，即分类的类别数。在tf_flowers数据集的情况下，这应该是5（花的种类）。
# activation="softmax"：使用softmax激活函数
output = keras.layers.Dense(n_classes, activation="softmax")(avg)

# 创建新模型
# 创建一个新的 Keras 模型，指定输入为预训练模型的输入，输出为添加的新层
model = keras.Model(inputs=base_model.input, outputs=output)
# 打印模型的结构和参数详情
model.summary()

# 设置学习率计划
# keras.optimizers.schedules.ExponentialDecay 设置一个指数衰减学习率，随着训练步数增加，学习率将逐步减小
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.02,     # 初始学习率 0.02
    decay_steps=10000,      # 每10000步学习率进行一次衰减
    decay_rate=0.9)     # 衰减率为0.9
# 遍历预训练模型的所有层，并将它们设置为不可训练（trainable=False）
# 为了保留这些层已经学到的特征，并只训练我们添加的新层
for layer in base_model.layers:
  layer.trainable = False
# 优化器 SGD
# learning_rate=lr_schedule：使用上面定义的指数衰减学习率。
# momentum=0.9：动量设置为0.9，可以帮助加速SGD在相关方向上的收敛，并抑制震荡。
optimizer = keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9)
# 配置优化器、损失函数、以准确率作为判断标准
model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
# 开始训练
history = model.fit(train_set, epochs=5, steps_per_epoch=dataset_size*0.75//batch_size, validation_data=valid_set, validation_steps = dataset_size*0.15//batch_size)