#!/usr/bin/python3
#this script is used for build a AlexNet to classifier cifar-10

import pandas as pd
import numpy as np
import tensorflow as tf

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)
def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch


(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
X_train = X_train.astype(np.float32) / 255.0
X_test = X_test.astype(np.float32) / 255.0
y_train = y_train.astype(np.int32)
y_train=y_train.reshape([-1,])
y_test = y_test.astype(np.int32)
y_test=y_test.reshape([-1,])
rnd_idx_2500 = np.random.permutation(len(X_train))[500:3000]
rnd_idx_500 =np.random.permutation(len(X_train))[0:500]
X_valid, X_train = X_train[rnd_idx_500], X_train[rnd_idx_2500]
y_valid, y_train = y_train[rnd_idx_500], y_train[rnd_idx_2500]
X_test=X_test[np.random.permutation(len(X_test))[0:500]]
y_test=y_test[np.random.permutation(len(X_test))[0:500]]


reset_graph()
X = tf.placeholder(tf.float32, shape=[None, 32,32,3],name="X")
y = tf.placeholder(tf.int32, shape=[None],name="y")
n_epochs=100
batch_size=64

with tf.name_scope("alexnet"):
  conv1=tf.layers.conv2d(X, filters=64, kernel_size=[5,5],strides=[1,1],padding='VALID',name="conv1",activation=tf.nn.relu)
  pool2=tf.layers.max_pooling2d(conv1, pool_size=[3,3], strides=[2,2],padding="VALID",name='pool2')
  conv3=tf.layers.conv2d(pool2, filters=64, kernel_size=[5,5],strides=[1,1],padding='VALID',activation=tf.nn.relu,name="conv3")
  pool4=tf.layers.max_pooling2d(conv3, pool_size=[3,3], strides=[2,2],padding="VALID",name='pool4')
  pool4_flat=tf.layers.flatten(pool4)
  fcl5=tf.layers.dense(pool4_flat, 384,activation=tf.nn.relu,name="fcl5")
  fcl6=tf.layers.dense(fcl5, 192,activation=tf.nn.relu,name='fcl6')
  logits=tf.layers.dense(fcl6, 10,name='fcl7')
with tf.name_scope("loss"):
  xentropy=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
  loss=tf.reduce_mean(xentropy)
with tf.name_scope("train"):
  optimizer=tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.9)
  training_op=optimizer.minimize(loss)
with tf.name_scope("eval"):
  correct=tf.nn.in_top_k(logits, y, 1)
  accuracy=tf.reduce_mean(tf.cast(correct,tf.float32))
init=tf.global_variables_initializer()
saver=tf.train.Saver()

with tf.Session() as sess:
  init.run()
  for epoch in range(n_epochs):
    for X_batch,y_batch in shuffle_batch(X_train, y_train, batch_size):
      sess.run(training_op,feed_dict={X:X_batch,y:y_batch})
    if (epoch+1)%10 == 0 :
      acc_val=accuracy.eval(feed_dict={X:X_valid,y:y_valid})
      print("epoch is : ",epoch+1,"accuracy is :",acc_val)
  save_path=saver.save(sess, "./alexnet_final_model.ckpt")
with tf.Session() as sess:
  saver.restore(sess, "./alexnet_final_model.ckpt")
  test_acc_val=accuracy.eval(feed_dict={X:X_test,y:y_test})
  print("test_dataset accuracy is :",test_acc_val)

##use keras to build AlexNet
import keras
from keras.models import Sequential
model=Sequential()
from keras.layers import Dense,Conv2D,MaxPool2D,Flatten
from keras.layers import Activation

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
X_train = X_train.astype(np.float32) / 255.0
X_test = X_test.astype(np.float32) / 255.0
X_valid, X_train = X_train[:500], X_train[500:3000]
y_valid, y_train = y_train[:500], y_train[500:3000]
X_test=X_test[:500]
y_test=y_test[:500]
onehot_train = keras.utils.to_categorical(y_train, num_classes=10)
onehot_test=keras.utils.to_categorical(y_test,num_classes=10)


model.add(Conv2D(64,(5,5),input_shape=(32,32,3)))
model.add(Activation('relu'))
model.add(MaxPool2D([3,3],2))
model.add(Conv2D(64,(5,5),activation='relu'))
model.add(MaxPool2D([3,3],2))
model.add(Flatten())
model.add(Dense(384,activation='relu'))
model.add(Dense(192,activation='relu'))
model.add(Dense(10,activation='softmax'))
model.summary()
from keras.utils.vis_utils import plot_model
plot_model(model,to_file="keras_alexnet.png")

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
SVG(model_to_dot(model).create(prog="dot",format='svg'))

from keras import optimizers
sgd=optimizers.SGD(lr=0.01,momentum=0.9)
model.compile(optimizer=sgd,loss="categorical_crossentropy",metrics=['accuracy'])
model.fit(x=X_train,y=onehot_train,epochs=50,batch_size=20)
model.evaluate(X_test,onehot_test,batch_size=128)


