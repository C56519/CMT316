import numpy as np
from sklearn.datasets import make_moons
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.svm import SVR

# 1 SVM for classification

# 1.1 封装两个函数用于绘
# 绘制原始数据和坐标轴
def plot_dataset(X, y, axes):
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], "bs")
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "g^")
    plt.axis(axes)
    plt.grid(True, which='both')
    plt.xlabel(r"$x_1$", fontsize=20)
    plt.ylabel(r"$x_2$", fontsize=20, rotation=0)
# 绘制决策边界
def plot_predictions(clf, axes):
    x0s = np.linspace(axes[0], axes[1], 100)
    x1s = np.linspace(axes[2], axes[3], 100)
    x0, x1 = np.meshgrid(x0s, x1s)
    X = np.c_[x0.ravel(), x1.ravel()]
    y_pred = clf.predict(X).reshape(x0.shape)
    y_decision = clf.decision_function(X).reshape(x0.shape)
    plt.contourf(x0, x1, y_pred, cmap=plt.cm.brg, alpha=0.2)
    plt.contourf(x0, x1, y_decision, cmap=plt.cm.brg, alpha=0.1)

# 1.2 准备数据集
# make_moons() 随机的二分类数据集，是 sklearn 库中的一个函数，用于生成具有两个半月形状的数据集，参数：
# n_samples：生成的样本数量，默认为 100。
# noise：生成的数据中的噪声级别，默认为 0.15。噪声越大，数据点越分散。
# random_state：随机种子，用于生成随机数据以确保可重复性。
X_svm_clf, Y_svm_clf = make_moons(n_samples=100, noise=0.15, random_state=42)

# 1.3 训练
# 使用 pipline 的两种方法
# 首先，特征缩放；然后使用 SVM
svm_clf_poly100 = Pipeline([
    # 特征缩放：StandardScaler()是标准化特征值的方法，它将数据进行标准化处理，使得每个特征的平均值为 0，方差为 1。
    ("scalar", StandardScaler()),
    # SVM
    # kernel="poly": 使用多项式核函数；
    # degree=10: 多项式核函数的次数为 10；
    # coef0=100: 多项式核函数的独立项系数为 100；
    # C=5: 惩罚程度为 5
    ("svm_clf", SVC(kernel="poly", degree=10, coef0=100, C=5))
])
svm_clf_poly100.fit(X_svm_clf, Y_svm_clf)
# 再训练一个模型，上一个模型使用了高次数的多项式核函数 degree，并设置了较大的独立项系数 coef0
# 这个模型使用了较低次数的多项式核函数 degree，设置了较小的独立项系数 coef0
# 使用了mark_pipeline()函数来创建pipline，然后对比一下结果
svm_clf_poly1 = make_pipeline(StandardScaler(),
                                    SVC(kernel="poly", degree=3, coef0=1, C=5))

svm_clf_poly1.fit(X_svm_clf, Y_svm_clf)

# 1.4 画图，这次是两个图并排展示结果
fig, axes = plt.subplots(ncols=2, figsize=(10.5, 4), sharey=True)

plt.sca(axes[0])
plot_predictions(svm_clf_poly100, [-1.5, 2.45, -1, 1.5])
plot_dataset(X_svm_clf, Y_svm_clf, [-1.5, 2.4, -1, 1.5])
plt.title(r"$d=3, r=1, C=5$", fontsize=18)

plt.sca(axes[1])
plot_predictions(svm_clf_poly1, [-1.5, 2.45, -1, 1.5])
plot_dataset(X_svm_clf, Y_svm_clf, [-1.5, 2.4, -1, 1.5])
plt.title(r"$d=10, r=100, C=5$", fontsize=18)
plt.ylabel("")

plt.show()


# 2 SVM 参数调整, 也称为调整 超参数


# 在SVM用于分类的 SVC(kernel, gamma, C) 函数中有的三个参数
# kernel = "String"，选择SVM的核函数，用于将输入数据映射到更高维的特征空间。可选值：{"linear", "poly", "rbf", "sigmoid", "precomputed"}
# gamma = float，核函数的系数，控制了单个训练样本对决策边界的影响程度影响了模型的复杂度和决策边界的形状，越大边界越平滑
# C = float，正则化参数，控制了模型对训练数据的拟合程度

# 2.1 方案一：以 gamma 和 C 这两个参数为例
gamma1, gamma2 = 0.1, 5
C1, C2 = 0.001, 1000
# (1) 定义一个元组，可以省略元组最外围的括号
hyperparams = (gamma1, C1), (gamma1, C2), (gamma2, C1), (gamma2, C2)
svm_clf_hyperparams1 = []
# 遍历，每个都训练一次
for gamma, C in hyperparams:
    svm_clf_choosing = Pipeline([
            ("scaler", StandardScaler()),
            ("svm_clf", SVC(kernel="rbf", gamma=gamma, C=C))
        ])
    svm_clf_choosing.fit(X_svm_clf, Y_svm_clf)
    # 添加本次训练的模型到列表中
    svm_clf_hyperparams1.append(svm_clf_choosing)

# (2) 画图，比较每个的结果
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10.5, 7), sharex=True, sharey=True)
for i, svm_clf in enumerate(svm_clf_hyperparams1):
    plt.sca(axes[i // 2, i % 2])
    plot_predictions(svm_clf, [-1.5, 2.45, -1, 1.5])
    plot_dataset(X_svm_clf, Y_svm_clf, [-1.5, 2.45, -1, 1.5])
    gamma, C = hyperparams[i]
    plt.title(r"$\gamma = {}, C = {}$".format(gamma, C), fontsize=16)
    if i in (0, 1):
        plt.xlabel("")
    if i in (1, 3):
        plt.ylabel("")

plt.show()

# 2.2 方案二：使用 sklearn 的网格搜索来选合适的参数，这次调三个参数
# (1) 定义搜索空间
parameters = {'kernel':('linear', 'rbf'), 'gamma':[0.1, 5], 'C':[0.001, 1, 10, 1000]}

# (2) 定义分类模型
model = SVC(kernel="rbf")

# (3) 定义交叉验证方法
# 定义了重复的分层K折交叉验证。这里设置了5折交叉验证，重复3次。
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)

# (4) 配置搜索参数
# 参数包括模型对象、超参数空间、评分指标、并行处理的工作数量和交叉验证方法。
search = GridSearchCV(model, parameters, scoring='accuracy', n_jobs=-1, cv=cv)

# (5) 开始搜索
result = search.fit(X_svm_clf, Y_svm_clf)
# 结果
print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)


# 3 SVM for regression

# 3.1 定义一个用于画图的函数
def plot_svm_regression(svm_reg, X, y, axes):
    x1s = np.linspace(axes[0], axes[1], 100).reshape(100, 1)
    y_pred = svm_reg.predict(x1s)
    plt.plot(x1s, y_pred, "k-", linewidth=2, label=r"$\hat{y}$")
    plt.plot(x1s, y_pred + svm_reg.epsilon, "k--")
    plt.plot(x1s, y_pred - svm_reg.epsilon, "k--")
    plt.scatter(X[svm_reg.support_], y[svm_reg.support_], s=180, facecolors='#FFAAAA')
    plt.plot(X, y, "bo")
    plt.xlabel(r"$x_1$", fontsize=18)
    plt.legend(loc="upper left", fontsize=18)
    plt.axis(axes)

# 3.2 数据集
np.random.seed(42)
m = 100
X_svm_reg = 2 * np.random.rand(m, 1) - 1
Y_svm_reg = (0.2 + 0.1 * X_svm_reg + 0.5 * X_svm_reg**2 + np.random.randn(m, 1)/10).ravel()

# 3.3 训练
# SVR(kernel, degree, C, epsilon, gamma)，参数：
# kernel="String", "linear"：线性核函数。"poly"：多项式核函数。"rbf"：径向基函数（RBF）核函数。"sigmoid"：Sigmoid 核函数。
# degree=int, 多项式核函数的次数。仅当 kernel 参数设置为 "poly" 时有效
# C：正则化参数
# epsilon：ε参数, 指定了在拟合过程中所允许的目标变量的偏差，默认值为 0.1。
# gamma：核函数的系数。仅当 kernel 参数设置为 "rbf"、"poly" 或 "sigmoid" 时有效。它定义了核函数的影响范围
svm_reg1 = SVR(kernel="poly", degree=2, C=100, epsilon=0.1, gamma="scale")
svm_reg2 = SVR(kernel="poly", degree=2, C=0.01, epsilon=0.1, gamma="scale")
svm_reg1.fit(X_svm_reg, Y_svm_reg)
svm_reg2.fit(X_svm_reg, Y_svm_reg)

# 3.4 画图
fig, axes = plt.subplots(ncols=2, figsize=(9, 4), sharey=True)
plt.sca(axes[0])
plot_svm_regression(svm_reg1, X_svm_reg, Y_svm_reg, [-1, 1, 0, 1])
plt.title(r"$degree={}, C={}, \epsilon = {}$".format(svm_reg1.degree, svm_reg1.C, svm_reg1.epsilon), fontsize=18)
plt.ylabel(r"$y$", fontsize=18, rotation=0)
plt.sca(axes[1])
plot_svm_regression(svm_reg2, X_svm_reg, Y_svm_reg, [-1, 1, 0, 1])
plt.title(r"$degree={}, C={}, \epsilon = {}$".format(svm_reg2.degree, svm_reg2.C, svm_reg2.epsilon), fontsize=18)

plt.show()