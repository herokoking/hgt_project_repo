#!/usr/bin/python3
#this script is for chapter8 exercise 9 and 10
#exercise 9
from sklearn.datasets import fetch_mldata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
mnist = fetch_mldata('mnist-original', data_home='./datasets/')
X, y = mnist.data, mnist.target

from sklearn.model_selection import train_test_split,GridSearchCV
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=10000,random_state=42)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
rnd_clf=RandomForestClassifier(n_estimators=100,max_leaf_nodes=10,n_jobs=-1)
rnd_clf.fit(X_train,y_train)
y_pred=rnd_clf.predict(X_test)
print(accuracy_score(y_test, y_pred))

#PCA
from sklearn.decomposition import PCA,KernelPCA
from sklearn.manifold import LocallyLinearEmbedding

pca=PCA(n_components=0.95)
X_train_reduce=pca.fit_transform(X_train)
X_test_reduce=pca.transform(X_test)
rnd_clf.fit(X_train_reduce,y_train)
y_pred=rnd_clf.predict(X_test_reduce)
print(accuracy_score(y_test, y_pred))

#exercise 10 
#在60000内随机抽取10000个不重复的整数
np.random.seed(42)
m=10000
idx=np.random.permutation(60000)[:m]
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
tsne = TSNE(n_components=2, random_state=42)
X_reduced = tsne.fit_transform(X[:1000])
plt.figure(figsize=(13,10))
plt.scatter(X_reduced[:, 0], X_reduced[:,1], c=y[:1000], cmap="jet")
plt.axis('off')
plt.colorbar()
plt.show()


from sklearn.preprocessing import MinMaxScaler
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

def plot_digits(X, y, min_distance=0.05, images=None, figsize=(13, 10)):
    # Let's scale the input features so that they range from 0 to 1
    X_normalized = MinMaxScaler().fit_transform(X)
    # Now we create the list of coordinates of the digits plotted so far.
    # We pretend that one is already plotted far away at the start, to
    # avoid `if` statements in the loop below
    neighbors = np.array([[10., 10.]])
    # The rest should be self-explanatory
    plt.figure(figsize=figsize)
    cmap = mpl.cm.get_cmap("jet")
    digits = np.unique(y)
    for digit in digits:
        plt.scatter(X_normalized[y == digit, 0], X_normalized[y == digit, 1], c=[cmap(digit / 9)])
    plt.axis("off")
    ax = plt.gcf().gca()  # get current axes in current figure
    for index, image_coord in enumerate(X_normalized):
        closest_distance = np.linalg.norm(np.array(neighbors) - image_coord, axis=1).min()
        if closest_distance > min_distance:
            neighbors = np.r_[neighbors, [image_coord]]
            if images is None:
                plt.text(image_coord[0], image_coord[1], str(int(y[index])),
                         color=cmap(y[index] / 9), fontdict={"weight": "bold", "size": 16})
            else:
                image = images[index].reshape(28, 28)
                imagebox = AnnotationBbox(OffsetImage(image, cmap="binary"), image_coord)
                ax.add_artist(imagebox)


plot_digits(X_reduced, y)
#标示数字图像
plot_digits(X_reduced, y,images=X[:1000],figsize=(35,25))

##比较不同的降维方法效果(都降到2维),因为计算机配置问题，只取前1000个测试比较
#PCA
from sklearn.decomposition import PCA,KernelPCA
from sklearn.manifold import LocallyLinearEmbedding,TSNE

pca=PCA(n_components=2,random_state=42)
X_PCA_reduced=pca.fit_transform(X[:1000])

kpca=KernelPCA(n_components=2,random_state=42)
X_KPCA_reduced=kpca.fit_transform(X[:1000])

lle=LocallyLinearEmbedding(n_components=2,random_state=42)
X_lle_reduced=lle.fit_transform(X[:1000])

tsne=TSNE(n_components=2,random_state=42)
X_tsne_reduced=tsne.fit_transform(X[:1000])

for reduced in (X_PCA_reduced,X_KPCA_reduced,X_lle_reduced,X_tsne_reduced):
	plt.figure(figsize=(13,10))
	plt.scatter(reduced, y)
	plt.show()
	plt.close()
