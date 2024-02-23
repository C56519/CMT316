import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LogisticRegression

# 1 二元分类

# 1.1 加载鸢尾花数据集
# iris对象的 keys 为：['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module']
# 里面的每个key对应的value都是个列表
# feature_names: 也就是 feature, ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
# data: 里面是各个特征对应的数据 [6.9 3.1 5.4 2.1]
# target_names: 也就是 label, ['setosa' 'versicolor' 'virginica'], 也就是 [0, 1, 2] 将 virginica(2) 分类为1， 其余为0
iris = datasets.load_iris()
list(iris.keys())
print(iris.target)
print(iris.feature_names)

# 1.2 构建训练集
# iris['data']取iris对象的data键对应的value列表。[:, (2, 3)] 首先 : 表示选择整个列表的所有行，(2, 3) 表示将所有行的索引为2-3，也就是3-4列的所有数据，使用切片拷贝到X_logistic列表
X_logistic = iris['data'][:, (2, 3)]
# 取iris[‘target’] 列表，遍历每个元素，如果该元素为2，返回True，其余的返回False。然后，使用astype(np.int)遍历这些元素组成的新列表，将其转换为 int 型数据，也就是 0和1
# list.astype(dtype) 用于将列表中的元素转换为指定的数据类型，参数 dtype: 指定数据类型
Y_logisitc = (iris['target'] == 2).astype(int)

# 1.3 训练
# LogisticRegression() 逻辑回归
# 参数：
# penalty：正则化项，默认为 'l2'。可以是 'l1'、'l2' 或者 'none'，分别对应 L1 正则化、L2 正则化和无正则化。
# C：正则化强度的倒数，即正则化系数的倒数。较小的 C 表示更强的正则化。默认为 1.0。
# solver：用于求解优化问题的算法。默认为 'lbfgs'，可以是 'newton-cg'、'lbfgs'、'liblinear'、'sag' 或者 'saga'。每种算法都有其特定的优势和限制，具体选择可以根据数据集的大小和特性来决定。
# max_iter：最大迭代次数，默认为 100。用于控制求解优化问题的迭代次数。
# random_state：随机数种子，用于初始化模型参数以保证结果的可重复性。
log_reg = LogisticRegression(solver="liblinear", C=10**10, random_state=42)
log_reg.fit(X_logistic, Y_logisitc)

# 1.4 预测
# (1) 生成随机数据
# np.linspace() 生成了一个从 2.9 到 7 的等间隔的包含 500 个点的一维数组，表示 x0 轴的坐标点。生成了一个从 0.8 到 2.7 的等间隔的包含 200 个点的一维数组，表示 x1 轴的坐标点。
# np.reshape(-1, 1) 将一维数组转换为列向量的形式，因为 np.meshgrid() 函数需要的是一维数组。
# np.meshgrid() 将这两个一维数组作为参数，生成了一个[[x0, x1], [x0, x1]...] 二维的网格坐标矩阵
x0, x1 = np.meshgrid(
        np.linspace(2.9, 7, 500).reshape(-1, 1),
        np.linspace(0.8, 2.7, 200).reshape(-1, 1),
    )
# (2) 将多个一维数组按列合并成一个二维数组
# x0.ravel() 和 x1.ravel() 分别是将二维数组展平为一维数组得到的结果，然后通过 np.c_[] 方法将这两个一维数组按列连接起来，形成一个二维数组。
# 其中第一列是 x0.ravel() 中的元素，第二列是 x1.ravel() 中的元素。
X_logistic_predict = np.c_[x0.ravel(), x1.ravel()]
# (3) 预测
# 对网格中的每个点进行预测，并返回每个点属于正类和负类的概率。
Y_logistic_predict = log_reg.predict_proba(X_logistic_predict)
print(Y_logistic_predict)

# 1.5 画图
plt.figure(figsize=(10, 4))
# (1) 绘制原始数据样本点：将原始数据集中的样本点按类别（正类和负类）绘制到图上。
plt.plot(X_logistic[Y_logisitc==0, 0], X_logistic[Y_logisitc==0, 1], "bs")
plt.plot(X_logistic[Y_logisitc==1, 0], X_logistic[Y_logisitc==1, 1], "g^")

# (2) 绘制决策边界的等高线
# 获取逻辑回归模型预测的概率矩阵中，属于正类（类别为 1）的概率列
# 由于逻辑回归预测时返回的的概率矩阵有两列，第一列表示样本属于负类（类别为 0）的概率，第二列表示样本属于正类的概率。
# y_proba[:, 1] 选取了第二列的概率值。并使用.reshape(x0.shape)) 将一维的概率数组重新转换为与 x0 相同形状的二维数组，以便后续绘制等高线。
zz = Y_logistic_predict[:, 1].reshape(x0.shape)
# 绘制决策边界的等高线图。其中，x0 和 x1 表示决策边界上的坐标点，zz 表示每个坐标点对应的概率值，cmap=plt.cm.brg 表示使用蓝-红-绿的颜色映射。
contour = plt.contour(x0, x1, zz, cmap=plt.cm.brg)

# (3) 计算决策边界
# left_right = np.array([2.9, 7])：设置 x 轴坐标的范围。
left_right = np.array([2.9, 7])
# 计算决策边界：超平面方程, w0 +w1 * x1 + w2 * x2 = 0
boundary = -(log_reg.coef_[0][0] * left_right + log_reg.intercept_[0]) / log_reg.coef_[0][1]

# (4) 开始绘图
# 给等高线添加标签。contour 是之前绘制的等高线对象，inline=1 表示将标签内嵌到等高线中，fontsize=12 表示标签的字体大小。
plt.clabel(contour, inline=1, fontsize=12)
# 绘制决策边界, left_right 是 x 轴的范围，boundary 是决策边界的 y 值，"k--" 表示黑色虚线，linewidth=3 表示线宽为 3。
plt.plot(left_right, boundary, "k--", linewidth=3)
# 添加文本标签。分别在坐标点 (3.5, 1.5) 和 (6.5, 2.3) 处添加了文本 "Not Iris-Virginica" 和 "Iris-Virginica"，设置字体和颜色，ha="center" 表示水平居中对齐。
plt.text(3.5, 1.5, "Not Iris-Virginica", fontsize=14, color="b", ha="center")
plt.text(6.5, 2.3, "Iris-Virginica", fontsize=14, color="g", ha="center")
# 设置 x 轴和 y 轴的标签文字和字体
plt.xlabel("Petal length", fontsize=14)
plt.ylabel("Petal width", fontsize=14)
# 设置坐标轴的范围。指定了 x 轴的范围[2.9, 7] 和 y 轴的范围[0.8, 2.7]
plt.axis([2.9, 7, 0.8, 2.7])

plt.show()


# 2 多元分类


# target_names: 也就是 label, ['setosa' 'versicolor' 'virginica'], 也就是 [0, 1, 2] 将 其分为三类
# 2.1 训练集
# 对于 feature 列表，选择索引2，3号也就是第三四个，来进行训练
X_logistic_multi = iris['data'][:, (2, 3)]
Y_logisitc_multi = iris['target']

# 2.2 训练
log_reg_multi = LogisticRegression(multi_class="multinomial",solver="lbfgs", C=10, random_state=42)
log_reg_multi.fit(X_logistic_multi, Y_logisitc_multi)

# 2.3 预测
x0, x1 = np.meshgrid(
        np.linspace(0, 8, 500).reshape(-1, 1),
        np.linspace(0, 3.5, 200).reshape(-1, 1),
    )
X_logistic_multi_predict = np.c_[x0.ravel(), x1.ravel()]
# 每个点的各个分类的概率 和 预测值
Y_logisitc_multi_probability = log_reg_multi.predict_proba(X_logistic_multi_predict)
Y_logisitc_multi_predict = log_reg_multi.predict(X_logistic_multi_predict)
print("=====================================================")
print(Y_logisitc_multi_probability)
print(Y_logisitc_multi_predict)

# 2.4 绘图
zz1 = Y_logisitc_multi_probability[:, 1].reshape(x0.shape)
zz = Y_logisitc_multi_predict.reshape(x0.shape)

# (1) 创建窗口，并设置窗口尺寸
plt.figure(figsize=(10, 4))
# (2) 绘制原始数据集的散点图，分三个类别：Iris-Setosa、Iris-Versicolor 和 Iris-Virginica
# plt.plot(x, y, format_string, **kwargs)
# 绘制分类为 Iris-Virginica(Y_logisitc_multi==2) 的所有数据点
# 数据点的x坐标：X_logistic_multi[Y_logisitc_multi==2, 0]。先看里面，if 该数据点的lable为2，取feature列表中的第一列元素值，赋值给y坐标
# 数据点的y坐标：X_logistic_multi[Y_logisitc_multi==2, 1].先看里面，if 该数据点的lable为2, 取feature列表中的第二列元素值，赋值给y坐标
# 此外，配置 "g^" 表示使用绿色的三角形 (^) 标记绘制这些点。并标上label = Iris-Virginica 来语义化
plt.plot(X_logistic_multi[Y_logisitc_multi==2, 0], X_logistic_multi[Y_logisitc_multi==2, 1], "g^", label="Iris-Virginica")
plt.plot(X_logistic_multi[Y_logisitc_multi==1, 0], X_logistic_multi[Y_logisitc_multi==1, 1], "bs", label="Iris-Versicolor")
plt.plot(X_logistic_multi[Y_logisitc_multi==0, 0], X_logistic_multi[Y_logisitc_multi==0, 1], "yo", label="Iris-Setosa")

# (3) 绘制决策边界
from matplotlib.colors import ListedColormap
# 使用 ListedColormap 来用于给等高线图中的不同区域上色。
# ListedColormap() 函数接受一个颜色列表作为参数，其中每个颜色代表着不同的区域
custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])

# 1) 绘制等高线
#plt.contourf(X, Y, Z, levels, cmap)，参数说明：
# X 和 Y：定义了等高线网格的 x 和 y 坐标点的二维数组。
# Z：定义了等高线图的高度值，通常是一个与 X 和 Y 对应的二维数组。
# levels：可选参数，用于指定等高线的高度值范围。如果不提供，则自动生成等高线。
# cmap：可选参数，用于指定填充区域的颜色映射。
# 这里，传入要预测数据的两个feature和预测值，然后给不同区域上色
plt.contourf(x0, x1, zz, cmap=custom_cmap)
# 2) 在绘制好的等高线图上添加文字：概率值
contour = plt.contour(x0, x1, zz1, cmap=plt.cm.brg)
plt.clabel(contour, inline=1, fontsize=12)

# (4) 绘制坐标轴
# 在坐标轴上显示文字
plt.xlabel("Petal length", fontsize=14)
plt.ylabel("Petal width", fontsize=14)
# 添加图例，以便清楚地识别出图中不同元素所代表的含义。显示位置位置为 "center left"，字体大小为 14
plt.legend(loc="center left", fontsize=14)
# 设置 x 轴和 y 轴的范围为 [0, 7] 和 [0, 3.5]
plt.axis([0, 7, 0, 3.5])

plt.show()
