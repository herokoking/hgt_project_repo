#!/usr/bin/python3
#this script is used for build a VGG_Net to classifier cifar-10

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

def vgg_block(inputs,num_convs,out_depth,scope="vgg_block",reuse=None):
  in_depth=inputs.get_shape().as_list()[-1]
  with tf.variable_scope(scope,reuse=reuse):
    net=inputs
    for i in range(num_convs):
      var_name="conv%d"%i
      net=tf.layers.conv2d(net, filters=out_depth, kernel_size=3,strides=1,padding='SAME',activation=tf.nn.relu,name=var_name)
    net = tf.layers.max_pooling2d(net, pool_size=2, strides=2,name='max_pool')
    return net
def vgg_stack(inputs,num_convs,out_depths,scope="vgg_stack",reuse=None):
  with tf.variable_scope(scope,reuse=reuse):
    net=inputs
    for i,(n,d) in enumerate(zip(num_convs,out_depths)):
      net=vgg_block(net, num_convs=n, out_depth=d,scope="vgg_block%d"%i)
    return net
def vgg(inputs,num_convs,out_depths,num_outputs,scope="vgg",reuse=None):
  with tf.variable_scope(scope,reuse=reuse):
    net = vgg_stack(inputs, num_convs, out_depths)
    with tf.variable_scope("full_connect"):
      net=tf.layers.flatten(net)
      net=tf.layers.dense(net, 100,activation=tf.nn.relu,name='fc1')
      logits=tf.layers.dense(net, num_outputs,name='outputs')
      return logits

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
X_train = X_train.astype(np.float32) / 255.0
X_test = X_test.astype(np.float32) / 255.0
y_train = y_train.astype(np.int32)
y_train=y_train.reshape([-1,])
y_test = y_test.astype(np.int32)
y_test=y_test.reshape([-1,])
X_valid, X_train = X_train[:50], X_train[50:300]
y_valid, y_train = y_train[:50], y_train[50:300]
X_test=X_test[:50]
y_test=y_test[:50]


reset_graph()
X = tf.placeholder(tf.float32, shape=[None, 32,32,3],name="X")
y = tf.placeholder(tf.int32, shape=[None],name="y")
n_epochs=50
batch_size=20

with tf.name_scope("vgg"):
  logits=vgg(X, [1,1,2,2,2], [64,128,256,512,512], 10)
with tf.name_scope("loss"):
  xentropy=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
  loss=tf.reduce_mean(xentropy)
with tf.name_scope("train"):
  optimizer=tf.train.AdamOptimizer(learning_rate=0.01)
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



##use keras to build VGGNet
import keras
import tensorflow as tf
import pandas as pd
import numpy as np
from keras.models import Sequential
model=Sequential()
from keras.layers import Dense,Conv2D,MaxPool2D,Flatten
from keras.layers import Activation

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
X_train = X_train.astype(np.float32) / 255.0
X_test = X_test.astype(np.float32) / 255.0
X_valid, X_train = X_train[:50], X_train[50:300]
y_valid, y_train = y_train[:50], y_train[50:300]
X_test=X_test[:50]
y_test=y_test[:50]
onehot_train = keras.utils.to_categorical(y_train, num_classes=10)
onehot_test=keras.utils.to_categorical(y_test,num_classes=10)

model.add(Conv2D(64,(3,3),input_shape=(32,32,3)))
model.add(Activation('relu'))
model.add(Conv2D(64,(3,3),padding='same',activation='relu'))
model.add(MaxPool2D(pool_size=(2,2),padding='same'))
model.add(Conv2D(128,(3,3),padding='same',activation='relu'))
model.add(Conv2D(128,(3,3),padding='same',activation='relu'))
model.add(MaxPool2D(pool_size=(2,2),padding='same'))
model.add(Conv2D(256,(3,3),padding='same',activation='relu'))
model.add(Conv2D(256,(3,3),padding='same',activation='relu'))
model.add(Conv2D(256,(3,3),padding='same',activation='relu'))
model.add(MaxPool2D(pool_size=(2,2),padding='same'))
model.add(Conv2D(512,(3,3),padding='same',activation='relu'))
model.add(Conv2D(512,(3,3),padding='same',activation='relu'))
model.add(Conv2D(512,(3,3),padding='same',activation='relu'))
model.add(MaxPool2D(pool_size=(2,2),padding='same'))
model.add(Conv2D(512,(3,3),padding='same',activation='relu'))
model.add(Conv2D(512,(3,3),padding='same',activation='relu'))
model.add(Conv2D(512,(3,3),padding='same',activation='relu'))
model.add(MaxPool2D(pool_size=(2,2),padding='same'))
model.add(Flatten())
model.add(Dense(100,activation='relu'))
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
