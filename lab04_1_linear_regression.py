import numpy as np
import os   # 提供了许多函数来与操作系统交互

# np.random.seed(n) 设置n后，保证之后np.random产生的随机数都是固定的，比如设置42，表明第42堆种子，从而使实验结果可复现
np.random.seed(42)

# %matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

import warnings     # 导入警告库，并过滤掉特定的警告信息
warnings.filterwarnings(action="ignore", message="^internal gelsd")

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge


# Linear Regression


# 1 sklearn 线性回归
# 1.1 设置训练集
# np.random.rand(d0, d1, ...) 生成多维度的列表，参数设置列表的维度，列表内元素取值是从 [0, 1) 符合均匀分布的数据中随机抽取
# 如下：生成一个100行1列的列表，前面乘了2来放缩各个元素数值到两倍
X_reg = 2 * np.random.rand(100, 1)
# np.random.randn(d0, d1, ...) 同rand，一样，但是元素的取值是从 [0,1) 符合高斯分布(标准正态分布) 的数据中随机抽取
# yi = w * xi + b + noise, 这里w = 3. b = 4, 再加上一些标准正态分布噪声
Y_reg = 4 + 3 * X_reg + np.random.randn(100, 1)

# 1.2 训练
lin_reg = LinearRegression()
lin_reg.fit(X_reg, Y_reg)
lin_reg.intercept_, lin_reg.coef_

# 1.3 预测
# np.array(list) 将参数的list,object，元组等类列表对象转换成numpy数组
# 下面转换成一个2行1列的二维数组
X_reg_predict = np.array([[0], [2]])
# np.ones((2, 1)) 生成一个2行1列的元素值全为1的列表
Y_reg_gold = np.c_[np.ones((2, 1)), X_reg_predict]
lin_reg.predict(X_reg_predict)



# 2 sklearn 中的 SGDRegressor 随机梯度下降进行线性回归
# 2.1 配置训练集
sgd_reg = SGDRegressor(max_iter=50, penalty=None, eta0=0.1, random_state=42)
sgd_reg.fit(X_reg, Y_reg.ravel())
sgd_reg.intercept_, sgd_reg.coef_



# 3 线性回归，多项式回归综合 + 正则化
# 3.1 新数据集
np.random.seed(42)
X_new = 3 * np.random.rand(20, 1)
Y_new = 1 + 0.5 * X_new + np.random.randn(20, 1) / 1.5
# np.linspace(start, stop, num=50) 用于创建一个等间隔的一维数组
# 参数：start：起始索引; stop：结束索引; num：要生成的样本数，默认为 50
# return: 返回一个包含从 start 到 stop 之间的等间隔数字的一维数组。数组中的样本点数量由 num 参数指定
# np.reshape(arrary, newshape) 重新构造数组的维度
# 参数：arrary 要改变的数组；newshape 想要的维度
# 返回值函数返回一个按照数组的元素在内存中的存储顺序重新组织数据的具有新形状的数组，而不会更改原始数组的数据
X_new_predict = np.linspace(0, 3, 100).reshape(100, 1)

# 3.2 定义一个函数，完成这两个回归，并加入正则化 + 画图
# 参数
# regulaization_name: 用于传入的不同回归名，函数内部来进行指定的回归模型
# if_polynomial: 是否使用多项式特征扩展
# alphas: 一个列表，用于sklearn岭回归等函数的参数，来指定正则化强度，强度越大，正则化程度约强
# **model_kargs: python函数高级用法，属于函数构造器范畴，用于接受所有传入的未指定参数名的参数，形成一个字典
def train_and_plot_reg_or_polynomial_model(regulaization_name, if_polynomial, alphas, **model_kargs):
    # 使用 zip() 函数将 alphas 和样式字符串 ("b-", "g--", "r:") 一一对应，以便为每个不同的 alpha 值选择不同的样式来绘制曲线。
    # 以列表为例，zip(list1, list2) 函数接收两个列表，在每次循环中从两个列表中按照相同索引分别获取一个元素，然将这两个元素组合成一个元组，最后返回一个新的组合后的列表。相当于将这两个列表组合成一个列表
    for alpha, style in zip(alphas, ('b--', 'g--', 'r:')):
        # 如果 alpha 大于 0，则使用给定的 regulaization_name 和正则化参数值创建模型实例；否则使用普通的线性回归模型
        if alpha > 0:
            model = regulaization_name(alpha, **model_kargs)
        else: model = LinearRegression()
        # 如果使用多项式特征拓展， 则在训练模型之前添加了一个多项式特征扩展的预处理步骤
        if if_polynomial:
            # 创建了一个机器学习管道（Pipeline），在机器学习中，管道是一种将多个处理步骤组合在一起的方法，以便将其作为一个整体来处理数据。
            # 管道可以包含各种预处理步骤、特征提取、特征选择、模型拟合等步骤，使数据处理的流程更加清晰、简洁，并且可以更方便地进行交叉验证和参数调优。
            # 该管道：step1: 对原始特征进行多项式扩展，这里是转成10次多项式；step2: 对扩展后的特征进行标准化处理；step3: 对模型使用正则化
            model = Pipeline([
                    ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
                    ("std_scaler", StandardScaler()),
                    ("regul_reg", model)
                ])
        # 训练模型
        model.fit(X_new, Y_new)
        # 测试模型预测
        Y_new_predict = model.predict(X_new_predict)
        # 绘制图形
        if alpha > 0: linewidth = 2
        else: linewidth = 1
        # plt.plot(x, y, format_string, **kwargs) 用于绘制二维数据的折线图
        # 参数：x 要绘制所有数据点的横坐标数组；y 要绘制所有数据点的纵坐标数组；注意：当传入多组 x 和 y 时，它会将它们按照顺序绘制成多条曲线，每组 x 和 y 对应一条曲线
        # format_string 用于指定绘制的线条的样式(颜色、线型和标记)，是个字符串。如：'b-' 表示蓝色实线，'g--' 表示绿色虚线，'ro' 表示红色圆点
        # ** kwargs：可选的关键字参数，用于设置线条的其他属性，例如线宽、标签等。如：linewidth控制线条风格，label 这里用于显示不同曲线所使用的正则化强度
        plt.plot(X_new_predict, Y_new_predict, style, linewidth, label=r"$\alpha = {}$".format(alpha))

    plt.plot(X_new, Y_new, "b.", linewidth=3)
    plt.legend(loc="upper left", fontsize=15)
    plt.xlabel("$x_1$", fontsize=18)
    plt.axis([0, 3, 0, 4])

plt.figure(figsize=(8,4))
plt.subplot(121)
train_and_plot_reg_or_polynomial_model(Ridge, True, (0, 10, 100), random_state=42)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.subplot(122)
train_and_plot_reg_or_polynomial_model(Ridge, True, (0, 10**-5, 1), random_state=42)

plt.show()

"""
plt.plot(X_reg, Y_reg, "b.")
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.axis([0, 2, 0, 15])


plt.show()
"""