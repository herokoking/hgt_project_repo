#!/usr/bin/python3
# this script is used for classification about MNIST dataset
from sklearn.datasets import fetch_mldata
from sklearn import datasets
import numpy as np

mnist = fetch_mldata('mnist-original', data_home='./datasets/')
mnist
X, y = mnist.data, mnist.target
X.shape
y.shape
%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
one_digit = X[1]
one_digit_image = one_digit.reshape(28, 28)
plt.imshow(one_digit_image, cmap=matplotlib.cm.binary, interpolation="nearest")
plt.axis("off")
plt.show()

# 分割训练集和测试集
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
# 打乱顺序
import numpy as np
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]
y_train[1:5]

y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)
# fit a SGDClassifier
from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train)
sgd_clf.predict([one_digit])

# cross-validate to accuracy
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
skfolds = StratifiedKFold(n_splits=3, random_state=42)

for train_index, test_index in skfolds.split(X_train, y_train_5):
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train[train_index]
    y_train_folds = y_train_5[train_index]
    X_test_folds = X_train[test_index]
    y_test_folds = y_train_5[test_index]
    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_folds)
    n_correct = sum(y_pred == y_test_folds)
    print(n_correct / len(y_pred))

from sklearn.model_selection import cross_val_score
cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
confusion_matrix(y_train_5, y_train_pred)

precision_score(y_train_5, y_train_pred)
recall_score(y_train_5, y_train_pred)
f1_score(y_train_5, y_train_pred)
# 调用decision_function()方法返回每个实例的决策分数
y_scores = cross_val_predict(
    sgd_clf, X_train, y_train_5, cv=3, method="decison_function")
# 使用precision_recall_curve()计算所有可能的阈值的精度和召回率
from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--",
             label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
    plt.xlabel("Threshold", fontsize=16)
    plt.legend(loc="upper left", fontsize=16)
    plt.ylim([0, 1])

plt.figure(figsize=(8, 4))
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.xlim([-700000, 700000])
save_fig("precision_recall_vs_threshold_plot")
plt.show()


def plot_precision_vs_recall(precisions, recalls):
    plt.plot(recalls, precisions, "b-", linewidth=2)
    plt.xlabel("Recall", fontsize=16)
    plt.ylabel("Precision", fontsize=16)
    plt.axis([0, 1, 0, 1])

plt.figure(figsize=(8, 6))
plot_precision_vs_recall(precisions, recalls)
save_fig("precision_vs_recall_plot")
plt.show()

# ROC曲线
# 使用roc_curve()计算多种阈值的TPR和FPR
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)


def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)

plt.figure(figsize=(8, 6))
plot_roc_curve(fpr, tpr)
save_fig("roc_curve_plot")
plt.show()
# 计算曲线下的面积---auc值，用于比较不同的分类器性能
from sklearn.metrics import roc_auc_score
roc_auc_score(y_train_5, y_scores)
# 随机森林模型的得分
from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier(n_estimators=10, random_state=42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3,
                                    method="predict_proba")
y_scores_forest = y_probas_forest[:, 1]  # score = proba of positive class

fpr_forest, tpr_forest, thresholds_forest = roc_curve(
    y_train_5, y_scores_forest)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, "b:", linewidth=2, label="SGD")
plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
plt.legend(loc="lower right", fontsize=16)
save_fig("roc_curve_comparison_plot")
plt.show()

# compare two models auc_value
roc_auc_score(y_train_5, y_scores)
roc_auc_score(y_train_5, y_scores_forest)


## 多类别分类器
# 随机森林和朴素贝叶斯分类器是处理多类别的分类模型
# 而SVM和线性分类器则是严格的二元分类器
# 若使用二元分类器进行多类别分类任务时，它将自动转换OVA(one Vs rest)模式
# 亦可以强制指定OVA或者OVO
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
# 运用OVO + SGDClassifier
ovo_clf = OneVsOneClassifier(SGDClassifier(random_state=42))
ovo_clf.fit(X_train, y_train)
ovo_clf.predict([one_digit])

# 直接使用RandomForest
forest_clf.fit(X_train, y_train)
forest_clf.predict([one_digit])
forest_clf.predict_proba([one_digit])  # 查看该实例在各个类别的概率列表
# 使用交叉验证评估多类别分类器（和二元分类器一致）
cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy")

# 使用特征缩放后，再进行交叉验证
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaler = scaler.fit_transform(X_train.astype(np.float64))
cross_val_score(sgd_clf, X_train_scaler, y_train, cv=3, scoring="accuracy")

#错误分析
#多类别的混淆矩阵
y_train_pred=cross_val_predict(sgd_clf, X_train_scaler,y_train,cv=3)
conf_mx=confusion_matrix(y_train, y_train_pred)
plt.matshow(conf_mx,cmap=plt.cm.gray)
plt.show()      #呈现正确分类，越为光亮的格子，正确率越高

#变换为呈现错误的图像
row_sums=conf_mx.sum(axis=1,keepdims=True)
norm_conf_mx=conf_mx/row_sums
np.fill_diagonal(norm_conf_mx,0)
plt.matshow(norm_conf_mx,cmap=plt.cm.gray)
plt.show()      #越光亮的格子，错误率越高


## 多标签分类（输出多个二元标签的分类系统）
from sklearn.neighbors import KNeighborsClassifier      #KNN模型支持多标签分类
y_train_large=(y_train>=7)
y_train_odd=(y_train %2 ==1)
y_multilabel=np.c_[y_train_large,y_train_odd]
knn_clf=KNeighborsClassifier()
knn_clf.fit(X_train,y_multilabel)
knn_clf.predict([one_digit])
#经常采用多个标签得分相加后的平均值来评估模型
y_train_knn_pred=cross_val_predict(knn_clf, X_train,y_multilabel,cv=3)
f1_score(y_multilabel,y_train_knn_pred,average="weighted")  #按权重计算平均值
f1_score(y_multilabel, y_train_knn_pred,average='macro')    #直接计算算术平均值

## 多输出分类（每个实例会有多个标签，每个标签多个值）
noise=np.random.randint(0, 100, (len(X_train),784))
noise=np.random.randint(0, 100, (len(X_test),784))
