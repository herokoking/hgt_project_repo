#!/usr/bin/python3
#this script is used for build a GoogleNet(InceptionNet) to classifier cifar-10

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


#define inception block
def inception(x,d0_1,d1_1,d1_3,d2_1,d2_5,d3_1,name='inception',reuse=None):
  with tf.variable_scope(name,reuse=reuse):
    #define the first branch
    with tf.variable_scope('branch0'):
      branch0=tf.layers.conv2d(x, d0_1, [1,1],strides=1,padding='same')
      branch0=tf.layers.batch_normalization(branch0,training=is_training)
    with tf.variable_scope('branch1'):
      branch1=tf.layers.conv2d(branch0, d1_1, [1,1],strides=1,padding='same')
      branch1=tf.layers.conv2d(branch1, d1_3, [3,3],strides=1,padding='same')
      branch1=tf.layers.batch_normalization(branch1,training=is_training)
    with tf.variable_scope('branch2'):
      branch2=tf.layers.conv2d(branch1, d2_1, [1,1],strides=1,padding='same')
      branch2=tf.layers.conv2d(branch2, d2_5, [1,1],strides=1,padding='same')
      branch2=tf.layers.batch_normalization(branch2,training=is_training)
    with tf.variable_scope('branch3'):
      branch3=tf.layers.max_pooling2d(branch2, [3,3], strides=1,padding='same')
      branch3=tf.layers.conv2d(branch3, d3_1, [1,1],strides=1,padding='same')
      branch3=tf.layers.batch_normalization(branch3,training=is_training)
    net = tf.concat([branch0,branch1,branch2,branch3], axis=-1)
    return net
#use inception block to define googlenet
def googlenet(inputs,num_classes,is_training,reuse=None):
  with tf.variable_scope('googlenet',reuse=reuse):
    net=inputs
    with tf.variable_scope("block1"):
      net=tf.layers.conv2d(net, 64, [5,5],strides=2,name='conv5x5',padding='same')
    with tf.variable_scope("block2"):
      net=tf.layers.conv2d(net, 64, [1,1],strides=1,name='conv1x1',padding='same')
      net=tf.layers.conv2d(net, 192, [3,3],strides=1,name='conv3x3',padding='same')
      net=tf.layers.max_pooling2d(net, [3,3], strides=2,name='max_pool',padding='same')
    with tf.variable_scope('block3'):
      net=inception(net, 64, 96, 128, 16, 32, 32,name='inception1')
      net=inception(net, 128, 128, 192, 32, 96, 64,name='inception2')
      net=tf.layers.max_pooling2d(net, [3,3], strides=2,padding='same',name='max_pool')
    with tf.variable_scope("block4"):
      net=inception(net, 192, 96, 208, 16, 48, 64,name='inception1')
      net=inception(net, 160, 112, 224, 24, 64, 64,name='inception2')
      net=inception(net, 128, 128, 256, 24, 64, 64,name='inception3')
      net=inception(net, 112, 144, 288, 24, 64, 64,name='inception4')
      net=inception(net, 256, 160, 320, 32, 128, 128,name='inception5')
      net=tf.layers.max_pooling2d(net, [3,3], strides=2,padding='same',name='max_pool')
    with tf.variable_scope('block5'):
      net=inception(net, 256, 160, 320, 32, 128, 128,name='inception1')
      net=inception(net, 384, 182, 384, 48, 128, 128,name='inception2')
      net=tf.layers.average_pooling2d(net, [2,2], strides=2,padding='same',name='average_pool')
    with tf.variable_scope('classifier'):
      net=tf.layers.flatten(net)
      net=tf.layers.dense(net, num_classes,name='outputs')
    return net

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
is_training = tf.placeholder(tf.bool, [])

n_epochs=50
batch_size=20

with tf.name_scope("googlenet"):
  logits=googlenet(X,10,is_training)
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
      sess.run(training_op,feed_dict={X:X_batch,y:y_batch,is_training:True})
    if (epoch+1)%10 == 0 :
      acc_val=accuracy.eval(feed_dict={X:X_valid,y:y_valid,is_training:False})
      print("epoch is : ",epoch+1,"accuracy is :",acc_val)
  save_path=saver.save(sess, "./googlenet_final_model.ckpt")
with tf.Session() as sess:
  saver.restore(sess, "./googlenet_final_model.ckpt")
  test_acc_val=accuracy.eval(feed_dict={X:X_test,y:y_test,is_training:False})
  print("test_dataset accuracy is :",test_acc_val)



## use keras to build GoogleNet
from keras.models import Model
from keras.layers import Dense,BatchNormalization,Conv2D,MaxPool2D,AveragePooling2D,concatenate,Dropout,Input
import pandas as pd
import numpy as np
from keras.models import Sequential

model=Sequential()

def Conv2D_bn(x,filters,kernel_size,padding='same',strides=[1,1],name=None):
  if name is not None:
    bn_name=name+'bn'
    conv_name=name+'_conv'
  else:
    bn_name=None
    conv_name=None
  x=Conv2D(filters,kernel_size,padding=padding,strides=strides,activation='relu',name=conv_name)(x)
  x=BatchNormalization(axis=1,name=bn_name)(x)
  return x
def inception(x,filters):
  branch1=Conv2D_bn(x, filters, [1,1])
  branch2=Conv2D_bn(x, filters, [1,1])
  branch2=Conv2D_bn(branch2, filters, [3,3])
  branch3=Conv2D_bn(x, filters, [1,1])
  branch3=Conv2D_bn(branch3, filters, [5,5])
  branch4=MaxPool2D(pool_szie=(3,3),strides=(1,1),padding='same')(x)
  branch4=Conv2D_bn(branch4, filters, [1,1])
  x=concatenate([branch1,branch2,branch3,branch4],axis=1)
  return x

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

inpt = Input(shape=(32,32,3))

x=Conv2D_bn(inpt, 64, [7,7],strides=[2,2],padding='same')
x=MaxPool2D(pool_szie=[3,3],strides=[2,2],padding='same')(x)
x=Conv2D_bn(x, 192, [3,3])
x=MaxPool2D(pool_szie=[3,3],strides=[2,2],padding='same')(x)
x=inception(x, 64)
x=inception(x, 120)
x=MaxPool2D(pool_szie=[3,3],strides=[2,2],padding='same')(x)
x=inception(x, 128)
x=inception(x, 128)
x=inception(x, 132)
x=inception(x, 208)
x=inception(x, 256)
x=MaxPool2D(pool_szie=(3,3),strides=[2,2],padding='same')(x)
x=inception(x, 208)
x=inception(x, 256)
x=AveragePooling2D(pool_szie=[7,7],strides=[7,7],padding='same')(x)
x=Dropout(0.4)(x)
x=Dense(100,activation='relu')(x)
x=Dense(10,activation='softmax')(x)
model=Model(input=X_train,output=[x])
model.summary()
from keras.utils.vis_utils import plot_model
plot_model(model,to_file="keras_googlenet.png")
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
SVG(model_to_dot(model).create(prog="dot",format='svg'))

from keras import optimizers
sgd=optimizers.SGD(lr=0.01,momentum=0.9)
model.compile(optimizer=sgd,loss="categorical_crossentropy",metrics=['accuracy'])
model.fit(x=X_train,y=onehot_train,epochs=50,batch_size=20)
model.evaluate(X_test,onehot_test,batch_size=128)
