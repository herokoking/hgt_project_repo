#!/usr/bin/python3
import tensorflow as tf
import pandas as pd
import numpy as np

def hidden_layer(X,n_outputs,scope="hidden_layer",reuse=None):
	n_inputs=X.shape[-1]
	with tf.variable_scope(scope,reuse=reuse):
		w=tf.get_variable(initializer=tf.random_normal_initializer(),shape=(n_inputs,n_outputs),name="weights")
		b=tf.get_variable(initializer=tf.random_normal_initializer(),shape=(n_outputs),name="bias")
		net=tf.matmul(X,w)+b
	return net
def DNN(X,n_neurons_list,scope="DNN",reuse=None):
	net=X
	for i,n_neurons in enumerate(n_neurons_list):
		net=hidden_layer(net, n_neurons,scope="layer%d"%i,reuse=reuse)
		net=tf.nn.relu(net)
	net = hidden_layer(net, 1,scope="classifier_layer",reuse=reuse)
	net = tf.sigmoid(net)
	return net

dnn=DNN(X, [10,20,30])

loss_dnn=tf.losses.log_loss(labels=y, predictions=dnn)
learning_rate=0.01
optimizer=tf.train.GradientDescentOptimizer(lealearning_rate=learning_rate)
training_op=optimizer.minimize(loss)
init=tf.global_variables_initializer()
saver=tf.train.Saver()

with tf.Session() as sess:
	init.run()
	for e in range(2000):
		sess.run(training_op)
		if (e+1)%100==0:
			loss_val=loss_dnn.eval()
			print(value_val)
			saver.save(sess, "./model.ckpt",global_step=(e+1))

sess.close()


##模型恢复
#恢复模型结构
saver=tf.train.import_meta_graph("./model.ckpt-1000.meta")
#恢复模型参数
saver.restore(sess, "./model.ckpt-1000")


