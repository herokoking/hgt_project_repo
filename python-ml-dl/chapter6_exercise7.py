#!/usr/bin/python3
# this script is used for build a decision tree for satellites in chapter6 exercises7 and exercise8
# exercise7
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=10000, noise=0.4, random_state=42)
from sklearn.model_selection import train_test_split, GridSearchCV
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
print(X_train.shape)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

dtree = DecisionTreeClassifier(random_state=42)
max_leaf_nodes_list = list(range(2, 11))
parameters = {"max_leaf_nodes": max_leaf_nodes_list}
gc = GridSearchCV(dtree, param_grid=parameters, cv=10)
gc.fit(X_train, y_train)
# the best params
print(gc.best_params_)
# find out the best_dtree_model
best_dtree_model = gc.best_estimator_
y_pred = best_dtree_model.predict(X_test)
print(accuracy_score(y_test, y_pred))

# exercise8
from sklearn.model_selection import ShuffleSplit

n_trees = 1000
n_instances = 100

mini_sets = []

rs = ShuffleSplit(n_splits=n_trees, test_size=len(
    X_train) - n_instances, random_state=42)

for mini_train_index, mini_test_index in rs.split(X_train):
    X_mini_train = X_train[mini_train_index]
    y_mini_train = y_train[mini_train_index]
    mini_sets.append((X_mini_train, y_mini_train))

from sklearn.base import clone

forest = [clone(gc.best_estimator_) for _ in range(n_trees)]

accuracy_scores = []

for tree, (X_mini_train, y_mini_train) in zip(forest, mini_sets):
    tree.fit(X_mini_train, y_mini_train)

    y_pred = tree.predict(X_test)
    accuracy_scores.append(accuracy_score(y_test, y_pred))

np.mean(accuracy_scores)

# 建立一个1000*2000的df，每一行代表一个实例，每一列代表一个决策树模式预测的结果
Y_pred = np.empty([n_trees, len(X_test)], dtype=np.uint8)

for tree_index, tree in enumerate(forest):
    Y_pred[tree_index] = tree.predict(X_test)

from scipy.stats import mode

y_pred_majority_votes, n_votes = mode(Y_pred, axis=0)
y_pred_majority_votes=y_pred_majority_votes[0]
accuracy_score(y_test, y_pred_majority_votes)
