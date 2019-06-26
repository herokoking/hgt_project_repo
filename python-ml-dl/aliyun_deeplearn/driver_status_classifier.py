#!/usr/bin/python3
# this is the script uesd to import image and change np array
import numpy
import os
from PIL import Image  # 导入Image模块
from pylab import *  # 导入savetxt模块
import cv2


def get_imlist(path):  # 此函数读取特定文件夹下的jpg格式图像，返回图片所在路径的列表
  return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]


def image2ndarray(fdir_list):
  for i in range(len(fdir_list)):
    c = get_imlist(fdir_list[i])
    d = len(c)
    data = np.empty((d, 60, 80, 3))
    label_data=np.array([i]*d)
    while d > 0:
      img = cv2.imread(c[d - 1])
      b,g,r=cv2.split(img)
      img=cv2.merge([r,g,b])
      res = cv2.resize(img, dsize=(80, 60))
      img_ndarray = np.asarray(res, dtype='float16') / 255
      data[d - 1] = img_ndarray
      d = d - 1
    if i == 0:
      train_dataset = data
      train_label=label_data
    else:
      train_dataset = np.concatenate((train_dataset, data), axis=0)
      train_label=np.concatenate((train_label,label_data),axis=0)
  return train_dataset,train_label
os.chdir("./imgs/train/")
fdir_list=os.listdir("./")
train_dataset,train_label=image2ndarray(fdir_list)

rnd_idx = np.random.RandomState(seed=123).permutation(len(train_dataset))
X_train=train_dataset[rnd_idx]
y_train=train_label[rnd_idx]

X_test=X_train[0:50]
y_test=y_train[0:50]
X_valid, X_train = X_train[50:100], X_train[100:500]
y_valid, y_train = y_train[50:100], y_train[100:500]

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

reset_graph()
X = tf.placeholder(tf.float32, shape=[None, 60,80,3],name="X")
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
