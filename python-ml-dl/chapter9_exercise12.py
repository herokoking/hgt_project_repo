#!/usr/bin/python3
#this script is for chapter9 exercise 12
#exercise 9
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

def log_dir(prefix=""):
    now=datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir="tf_logs"
    if prefix:
        prefix += "-"
        name=prefix + "run-" + now
    return "{}/{}/".format(root_logdir,name)

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

def random_batch(X_train,y_train,batch_size):
    rnd_index=np.random.randint(0,len(X_train),batch_size)
    X_batch=X_train[rnd_index]
    y_batch=y_train[rnd_index]
    return X_batch,y_batch

def logistic_regression(X,y,initializer=None,seed=42,learning_rate=0.01):
    n_inputs_including_bias=int(X.get_shape()[1])
    with tf.name_scope("logistic_regression"):
        with tf.name_scope("model"):
            if initializer is None:
                initializer=tf.random_uniform([n_inputs_including_bias,1],-1,1,seed=seed)
            theta=tf.Variable(initializer,name="theta")
            logits=tf.matmul(X,theta,name="logits")
            y_proba=tf.sigmoid(logits)
        with tf.name_scope("train"):
            loss=tf.losses.log_loss(y,y_proba,scope="loss")
            optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
            training_op=optimizer.minimize(loss)
            loss_summary=tf.summary.scalar("loss",loss)
        with tf.name_scope("save"):
            saver=tf.train.Saver()
        with tf.name_scope("init"):
            init=tf.global_variables_initializer()
    return y_proba,loss,training_op,loss_summary,init,saver

reset_graph()
#add some features
m=1000
X_moons,y_moons=make_moons(m,noise=0.1,random_state=42)
#plt.plot(X_moons[y_moons==1,0],X_moons[y_moons==1,1],'go',label='Positive')
#plt.plot(X_moons[y_moons==0,0],X_moons[y_moons==0,1],'r^',label='Negative')
#plt.legend()
#plt.show()
X_moons_with_bias = np.c_[np.ones((m, 1)), X_moons]
y_moons_column_vector=y_moons.reshape(-1,1)
X_train,X_test,y_train,y_test=train_test_split(X_moons_with_bias,y_moons_column_vector,test_size=0.2,random_state=42)

X_train_enhanced = np.c_[X_train,
                         np.square(X_train[:, 1]),
                         np.square(X_train[:, 2]),
                         X_train[:, 1] ** 3,
                         X_train[:, 2] ** 3]
X_test_enhanced = np.c_[X_test,
                        np.square(X_test[:, 1]),
                        np.square(X_test[:, 2]),
                        X_test[:, 1] ** 3,
                        X_test[:, 2] ** 3]


n=X_train_enhanced.shape[1]
logdir = log_dir("logreg")
X=tf.placeholder(tf.float32,shape=(None,n),name="X")
y=tf.placeholder(tf.float32,shape=(None,1),name="y")
y_proba,loss,training_op,loss_summary,init,saver=logistic_regression(X,y)
file_writer=tf.summary.FileWriter(logdir,tf.get_default_graph())

n_epochs=1000
batch_size=50
checkpoint_path="./tmp/my_logreg_model.ckpt"
checkpoint_epoch_path=checkpoint_path+".epoch"
final_model_path="./tmp/my_logreg_final/model"

with tf.Session() as sess:
    for epoch in range(n_epochs):
        X_batch,y_batch=random_batch(X_train_enhanced,y_train,batch_size)
        #summary_str=loss_summary.eval(feed_dict={X:X_batch,y:y_batch})
        sess.run(training_op,feed_dict={X:X_batch,y:y_batch})
    best_theta=theta.eval()
    y_proba_val=y_proba.eval(feed_dict={X:X_test,y:y_test})
    saver.save(sess,final_model_path)
file_writer.close()