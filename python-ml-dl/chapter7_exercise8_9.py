#!/usr/bin/python3
#this script is for chapter7 exercise 8 and 9
#exercise 8
from sklearn.datasets import fetch_mldata
from sklearn import datasets
import numpy as np
mnist = fetch_mldata('mnist-original', data_home='./datasets/')
mnist
X, y = mnist.data, mnist.target

from sklearn.model_selection import train_test_split,GridSearchCV
X_training,X_test,y_training,y_test=train_test_split(X,y,test_size=10000,random_state=42)
X_train,X_validate,y_train,y_validate=train_test_split(X_training,y_training,test_size=10000,random_state=42)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float32))
X_test_scaled = scaler.transform(X_test.astype(np.float32))
X_validate_scaled=scaler.transform(X_validate.astype(np.float32))

from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,VotingClassifier
from sklearn.svm import LinearSVC,SVC
rnd_clf=RandomForestClassifier(n_estimators=100,max_leaf_nodes=10,n_jobs=-1)
extra_clf=ExtraTreesClassifier(n_estimators=100,max_leaf_nodes=10,n_jobs=-1)
lin_svc_clf=LinearSVC(C=1)
svc_clf=SVC(kernel='rbf',random_state=42,decision_function_shape='ovr',gamma='auto')

voting_hard_clf=VotingClassifier(estimators=[('randomForest',rnd_clf),('extratree',extra_clf),('liner_svm',lin_svc_clf),('svc_kernel',svc_clf)],voting='hard')
voting_soft_clf=VotingClassifier(estimators=[('randomForest',rnd_clf),('extratree',extra_clf),('liner_svm',lin_svc_clf),('svc_kernel',svc_clf)],voting='soft')
from sklearn.metrics import accuracy_score
for clf in (rnd_clf,extra_clf,lin_svc_clf,svc_clf,voting_hard_clf,voting_soft_clf):
	clf.fit(X_train_scaled, y_train)
	y_pred=clf.predict(X_validate_scaled)
	score=accuracy_score(y_validate, y_pred)
	print("validate_dataset:","classifier_name : ",clf.__class__.__name__,"accuracy_score : ",score)
	y_pred=clf.predict(X_test_scaled)
	score=accuracy_score(y_test, y_pred)
	print("test_dataset:","classifier_name : ",clf.__class__.__name__,"accuracy_score : ",score)



#exercise 9
estimators=[rnd_clf,extra_clf,lin_svc_clf]
X_val_predictions=np.empty((len(X_validate),3), dtype=np.float32)
for i,estimator in enumerate(estimators):
	X_val_predictions[:,i]=estimator.predict(X_validate_scaled)
rnd_second_clf=RandomForestClassifier(n_estimators=200,oob_score=True,random_state=42)
rnd_second_clf.fit(X_val_predictions,y_validate)
rnd_second_clf.oob_score_

X_test_predictions=np.empty((len(X_test_scaled),3), dtype=np.float32)
for i,estimator in enumerate(estimators):
	X_test_predictions[:,i]=estimator.predict(X_test_scaled)
y_pred=rnd_second_clf.predict(X_test_predictions)
accuracy_score(y_test, y_pred)
