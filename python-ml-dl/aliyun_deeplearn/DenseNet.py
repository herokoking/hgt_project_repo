#!/usr/bin/python3
#this script is used for build a DenseNet to classifier cifar-10

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

def bu_relu_conv(x,out_depth,scope='dense_basic_conv',reuse=None):
  with tf.variable_scope(scope,reuse=reuse):
    net=tf.layers.batch_normalization(x,training=is_training)
    net=tf.nn.relu(net)
    net=tf.layers.conv2d(net, out_depth, [3,3],strides=[1,1],activation=None,name='conv',padding='same')
    return net

def dense_block(x,growth_rate,num_layers,scope='dense_block',reuse=None):
  in_depth=x.get_shape().as_list()[-1]
  with tf.variable_scope(scope,reuse=reuse):
    net=x
    for i in range(num_layers):
      out=bu_relu_conv(net, growth_rate,scope='block%d' % i)
      net=tf.concat([net,out], axis=-1)
    return net
def transition(x,out_depth,scope='transition',reuse=None):
  in_depth=x.get_shape().as_list()[-1]
  with tf.variable_scope(scope,reuse=reuse):
    net=tf.layers.batch_normalization(x,training=is_training)
    net=tf.nn.relu(net)
    net=tf.layers.conv2d(net, out_depth, kernel_size=[1,1],strides=[1,1],activation=None,name='conv_t',padding='same')
    net=tf.layers.average_pooling2d(net, [2,2], [2,2],name='avg_pool')
    return net
def densenet(x,num_classes,growth_rate=32,block_layers=[6,12,24],scope='densenet',reuse=None):    
#block_layers设定的dense_block数量要根据图片长x宽决定，不可盲目定多，否则后面会报错，如果图片是112*112的可定标准的DenseNet-121，即block_layers=[6,12,24,16]
  with tf.variable_scope(scope,reuse=reuse):
    with tf.variable_scope('block0'):     
    #block0 的卷积层和最大池化层的步长要根据input_image的长宽而定，避免后面的transition中的平均池化层没法缩减图片尺寸
      net=tf.layers.conv2d(x, 64, kernel_size=[7,7],strides=1,padding='same',activation=None,name='conv_7x7')
      net=tf.layers.batch_normalization(net,training=is_training)
      net=tf.nn.relu(net)
      net=tf.layers.max_pooling2d(net, [3,3], strides=1,name='max_pool')
    for i,num_layers in enumerate(block_layers):
      count=i+1
      with tf.variable_scope('block'+str(count)):
        net=dense_block(net, growth_rate, num_layers)
        if i != len(block_layers)-1:
          current_depth=net.get_shape().as_list()[-1]
          net=transition(net, current_depth//2)
    with tf.variable_scope('classifier'):
      net=tf.layers.batch_normalization(net,training=is_training)
      net=tf.nn.relu(net)
      net=tf.layers.average_pooling2d(net, [2,2], strides=1,padding='valid')
      net=tf.layers.flatten(net)
      net=tf.layers.dense(net, num_classes,activation=None,name='outputs')
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
#X_train 的每张图为32*32

reset_graph()
X = tf.placeholder(tf.float32, shape=[None, 32,32,3],name="X")
y = tf.placeholder(tf.int32, shape=[None],name="y")
is_training = tf.placeholder(tf.bool, [])

n_epochs=50
batch_size=20

with tf.name_scope("DenseNet"):
  logits=densenet(X, 10)
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
  save_path=saver.save(sess, "./densenet_final_model.ckpt")
with tf.Session() as sess:
  saver.restore(sess, "./densenet_final_model.ckpt")
  test_acc_val=accuracy.eval(feed_dict={X:X_test,y:y_test,is_training:False})
  print("test_dataset accuracy is :",test_acc_val)

