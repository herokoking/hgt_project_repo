#!/usr/bin/python3
# this script is used for chapter13

import numpy as np
import pandas as pd
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

##use CNN to classifier mnist dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype(np.float32).reshape(-1, 28*28) / 255.0
X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
X_valid, X_train = X_train[:500], X_train[500:3000]
y_valid, y_train = y_train[:500], y_train[500:3000]
X_test=X_test[:500]
y_test=y_test[:500]


def variable_weight(shape,stddev=5e-2):
  init=tf.truncated_normal_initializer(stddev=stddev)
  return tf.get_variable(shape=shape,initializer=init,name="weight")
def variable_bias(shape):
  init=tf.constant_initializer(0.1)
  return tf.get_variable(shape=shape,initializer=init,name="bias")

def conv(x,ksize,out_depth,strides,padding='SAME',act=tf.nn.relu,scope="conv_layer",reuse=None):
  in_depth=x.get_shape().as_list()[-1]
  with tf.variable_scope(scope,reuse=reuse):
    shape=ksize+[in_depth,out_depth]
    with tf.variable_scope('kernel'):
      kernel=variable_weight(shape)
    strides=[1,strides[0],strides[1],1]
    conv=tf.nn.conv2d(x, kernel, strides, padding,name="conv")
    with tf.variable_scope("bias"):
      bias=variable_bias([out_depth])
    preact=tf.nn.bias_add(conv, bias)
    out=act(preact)
    return out
def max_pool(x,ksize,strides,padding='SAME',name='pool_layer'):
  return tf.nn.max_pool(x, [1,ksize[0],ksize[1],1], [1,strides[0],strides[1],1], padding,name=name)



reset_graph()
n_epochs = 10
batch_size = 100
X = tf.placeholder(tf.float32, shape=[None, 784],name="X")
X_reshape=tf.reshape(X, shape=[-1,28,28,1])
y= tf.placeholder(tf.int32, shape=[None],name="y")
with tf.name_scope("conv"):
	conv1=conv(X_reshape, [3,3], 32, [1,1],scope="conv_layer1")
	conv2=conv(conv1, [3,3], 64, [2,2],scope="conv_layer2")
with tf.name_scope("pool"):
	pool3=max_pool(conv2, [2,2], [2,2],padding='VALID')
	pool3_flat=tf.reshape(pool3, shape=[-1,64*7*7])
with tf.name_scope("full_connect"):
	fcl=tf.layers.dense(pool3_flat, 64,activation=tf.nn.relu,name="fcl")
	logits=tf.layers.dense(fcl,10,name='output')
	Y_prob=tf.nn.softmax(logits,name="Y_proba")

with tf.name_scope("loss"):
	xentropy=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
	loss=tf.reduce_mean(xentropy,name="loss")
with tf.name_scope("train"):
	optimizer=tf.train.AdamOptimizer(learning_rate=0.01)
	training_op=optimizer.minimize(loss=loss)
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
        acc_value=accuracy.eval(feed_dict={X:X_valid,y:y_valid})
        print("epoch is : ",epoch+1,"accuracy is :",acc_value)
    save_path=saver.save(sess,"./my_cnn_mnist_final_model.ckpt")

with tf.Session() as sess:
    saver.restore(sess,"./my_cnn_mnist_final_model.ckpt")
    test_acc_value=accuracy.eval(feed_dict={X:X_test,y:y_test})
    print("test_dataset accuracy is : ",test_acc_value)

##use advanture API to do the same thing CNN for MNIST datasets
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype(np.float32).reshape(-1, 28*28) / 255.0
X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
X_valid, X_train = X_train[:500], X_train[500:3000]
y_valid, y_train = y_train[:500], y_train[500:3000]
X_test=X_test[:500]
y_test=y_test[:500]

reset_graph()
n_epochs = 10
batch_size = 100
X = tf.placeholder(tf.float32, shape=[None, 784],name="X")
X_reshape=tf.reshape(X, shape=[-1,28,28,1])
y = tf.placeholder(tf.int32, shape=[None],name="y")
n_inputs=28*28
conv1_fmaps=32
conv1_ksize=[3,3]
conv1_stride=[1,1]
conv2_fmaps=64
conv2_ksize=[3,3]
conv2_stride=[2,2]

with tf.name_scope("cnn"):
	conv1=tf.layers.conv2d(X_reshape, filters=conv1_fmaps, kernel_size=conv1_ksize,strides=conv1_stride,padding='SAME',name='conv1',activation=tf.nn.relu)
	conv2=tf.layers.conv2d(conv1, filters=conv2_fmaps, kernel_size=conv2_ksize,strides=conv2_stride,padding='SAME',name='conv2',activation=tf.nn.relu)
	pool3=tf.layers.max_pooling2d(conv2,[2,2],[2,2],padding='VALID',name="max_pool_layer")
	pool3_flat=tf.reshape(pool3, shape=[-1,64*7*7])
	fcl=tf.layers.dense(pool3_flat, 64,activation=tf.nn.relu,name="fcl")
	logits=tf.layers.dense(fcl, 10,name='output')
with tf.name_scope("loss"):
	xentropy=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
	loss=tf.reduce_mean(xentropy)
with tf.name_scope("train"):
	optimizer=tf.train.AdamOptimizer(learning_rate=0.01)
	training_op=optimizer.minimize(loss=loss)
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
        acc_value=accuracy.eval(feed_dict={X:X_valid,y:y_valid})
        print("epoch is : ",epoch+1,"accuracy is :",acc_value)
    save_path=saver.save(sess,"./my_cnn_mnist_final_model.ckpt")

with tf.Session() as sess:
    saver.restore(sess,"./my_cnn_mnist_final_model.ckpt")
    test_acc_value=accuracy.eval(feed_dict={X:X_test,y:y_test})
    print("test_dataset accuracy is : ",test_acc_value)
