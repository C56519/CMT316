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



# 2 sklearn 中的 SGDRegressor 随机梯度下降进行线性回归， 并使用正则化
# 2.1 配置训练集
sgd_reg = SGDRegressor(max_iter=50, tol=np.infty, penalty=None, eta0=0.1, random_state=42)
sgd_reg.fit(X_reg, Y_reg)
sgd_reg.intercept_, sgd_reg.coef_


# 画图
plt.plot(X_reg, Y_reg, "b.")
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.axis([0, 2, 0, 15])


plt.show()