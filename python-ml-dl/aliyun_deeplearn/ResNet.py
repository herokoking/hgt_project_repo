#!/usr/bin/python3
#this script is used for build a ResNet to classifier cifar-10

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

#define subsample function
def subsample(x,factor,name=None):
  if factor==1:
    return x
  else:
    return tf.layers.max_pooling2d(x, [1,1], strides=factor,name=name,padding='same')

#define residual block
def residual_block(x,bottleneck_depth,out_depth,strides=1,name='residual_block'):
  net=x
  net=tf.layers.batch_normalization(net,training=is_training)
  net=tf.nn.relu(net)
  in_depth=net.get_shape().as_list()[-1]
  with tf.variable_scope(name):
    if in_depth==out_depth:
      shortcut=subsample(net, factor=strides,name='shortcut')
    else:
      shortcut=tf.layers.conv2d(net, out_depth, [1,1],strides=strides,name='shortcut',padding='same')
    residual=tf.layers.conv2d(net, bottleneck_depth, [1,1],strides=strides,activation=tf.nn.relu,name='conv1',padding='same')
    residual=tf.layers.conv2d(residual, bottleneck_depth, [3,3],strides=strides,activation=tf.nn.relu,name='conv2',padding='same')
    residual=tf.layers.conv2d(residual, out_depth, [1,1],strides=strides,name='conv3',padding='same')
    output=tf.nn.relu(shortcut+residual)
    return output
def resnet(inputs,num_classes,is_training=None,reuse=None):
  with tf.variable_scope('resnet',reuse=reuse):
    net=inputs
    with tf.variable_scope('block1'):
      net=tf.layers.conv2d(net, 32, [5,5],strides=2,activation=tf.nn.relu,padding='same',name='conv5x5')
    with tf.variable_scope('block2'):
      net=tf.layers.max_pooling2d(net, [3,3], 2, padding='same',name='max_pool')
      net=residual_block(net, 32, 128,name='residual_block1')
      net=residual_block(net, 32, 128,name='residual_block2')
    with tf.variable_scope('block3'):
      net=residual_block(net, 64, 256,name='residual_block1')
      net=residual_block(net, 64, 256,name='residual_block2')
    with tf.variable_scope('block4'):
      net=residual_block(net, 128, 512,name='residual_block1')
      net=residual_block(net, 128, 512,name='residual_block2')
    with tf.variable_scope('classifier_block'):
      net=tf.layers.batch_normalization(net,training=is_training)
      net=tf.layers.average_pooling2d(net, [2,2], strides=1,padding='valid')
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

with tf.name_scope("resnet"):
  logits=resnet(X,10,is_training)
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
  save_path=saver.save(sess, "./resnet_final_model.ckpt")
with tf.Session() as sess:
  saver.restore(sess, "./resnet_final_model.ckpt")
  test_acc_val=accuracy.eval(feed_dict={X:X_test,y:y_test,is_training:False})
  print("test_dataset accuracy is :",test_acc_val)
