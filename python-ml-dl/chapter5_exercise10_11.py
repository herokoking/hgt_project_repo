#!/usr/bin/python3
# this script is used for chapter5 exercise9,10
from sklearn.datasets import fetch_mldata
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
#exercise 9 
mnist = fetch_mldata('mnist-original', data_home='./datasets/')
mnist
X, y = mnist.data, mnist.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=10000, random_state=42)
from sklearn.svm import SVC, LinearSVC, SVR
lin_clf = LinearSVC(random_state=42)
lin_clf.fit(X_train, y_train)
y_pred = lin_clf.predict(X_train)
accuracy_score(y_train, y_pred)
# standerscaler first and then fit again
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float32))
X_test_scaled = scaler.transform(X_test.astype(np.float32))
lin_clf = LinearSVC(random_state=42)
lin_clf.fit(X_train_scaled, y_train)
y_pred = lin_clf.predict(X_train_scaled)
accuracy_score(y_train, y_pred)

# use SVC with kernel=RBF instead of LinearSVC
svm_clf = SVC(kernel='rbf', random_state=42,
              decision_function_shape='ovr', gamma='auto')
svm_clf.fit(X_train_scaled, y_train)
y_pred = svm_clf.predict(X_train_scaled)
accuracy_score(y_train, y_pred)


from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import reciprocal, uniform

param_distributions = {"gamma": reciprocal(0.001, 0.1), "C": uniform(1, 10)}
rnd_search_cv = RandomizedSearchCV(svm_clf, param_distributions, n_iter=10, verbose=2, cv=10,n_jobs=-1)
rnd_search_cv.fit(X_train_scaled[:1000], y_train[:1000])
svm_best_model = rnd_search_cv.best_estimator_
print(rnd_search_cv.best_params_)
svm_best_model.fit(X_train_scaled, y_train)
y_pred = svm_best_model.predict(X_train_scaled)
accuracy_score(y_train, y_pred)
y_pred = svm_best_model.predict(X_test_scaled)
accuracy_score(y_test, y_pred)



#exercise11

from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
X = housing["data"]
y = housing["target"]
X_train,X_test,y_train,y_test=train_test_split(X,y)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

from sklearn.svm import LinearSVR,SVR
from sklearn.metrics import mean_squared_error
lin_svr=LinearSVR(random_state=42)
lin_svr.fit(X_train_scaled, y_train)
y_pred=lin_svr.predict(X_train_scaled)
mean_squared_error(y_train, y_pred)

#RandomizedSearchCV
param_distributions = {"gamma": reciprocal(0.001, 0.1), "C": uniform(1, 10)}
rnd_search_cv=RandomizedSearchCV(SVR(), param_distributions,n_iter=10,verbose=2,cv=10,n_jobs=-1)
rnd_search_cv.fit(X_train_scaled,y_train)

print(rnd_search_cv.best_params_)
svr_best_model=rnd_search_cv.best_estimator_
y_pred=svr_best_model.predict(X_test_scaled)
mean_squared_error(y_true, y_pred)

