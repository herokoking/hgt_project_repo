#!/usr/bin/python3
#this script is for chapter10 exercise 9
#example in the book
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import os


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

def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch

n_inputs=28*28
n_hidden1=300
n_hidden2=100
n_outputs=10

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype(np.float32).reshape(-1, 28*28) / 255.0
X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
X_train = X_train[:5000]
y_train = y_train[:5000]



reset_graph()
#define X,y with placeholder
X=tf.placeholder(tf.float32,shape=(None,n_inputs),name="X")
y=tf.placeholder(tf.int32,name="y")
#define dnn 
with tf.name_scope("dnn"):
    hidden1=tf.layers.dense(X,n_hidden1,name="hidden1",activation=tf.nn.relu)
    hidden2=tf.layers.dense(hidden1,n_hidden2,name="hidden2",activation=tf.nn.relu)
    logits=tf.layers.dense(hidden2,n_outputs,name="outputs")
    y_proba=tf.nn.softmax(logits)
#define cost_fn
with tf.name_scope("loss"):
    xentropy=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=logits)
    loss=tf.reduce_mean(xentropy,name="loss")
#define Gradient_Descent_Optimizer
learning_rate=0.01
with tf.name_scope("train"):
    optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    training_op=optimizer.minimize(loss)
#define evalue the model
with tf.name_scope("eval"):
    correct=tf.nn.in_top_k(logits, y, 1)
    accuracy=tf.reduce_mean(tf.cast(correct,tf.float32))

#define init_part and saver
init=tf.global_variables_initializer()
saver=tf.train.Saver()

##run
n_epochs=20
batch_size=50
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for (X_batch,y_batch) in shuffle_batch(X_train, y_train, batch_size):
            training_op.run(feed_dict={X:X_batch,y:y_batch})
        print(accuracy.eval(feed_dict={X: X_batch, y: y_batch}))
    save_path = saver.save(sess, "./my_model_final.ckpt")


##use model to predict X_test
with tf.Session() as sess:
    saver.restore(sess,"./my_model_final.ckpt")
    X_new_scaled=X_test[:200]
    Z=y_proba.eval(feed_dict={X:X_new_scaled})
    y_pred=np.argmax(Z,axis=1)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test[:200],y_pred))




###exercise9

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt


n_inputs=28*28
n_hidden1=300
n_hidden2=100
n_outputs=10

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype(np.float32).reshape(-1, 28*28) / 255.0
X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
X_train = X_train[:5000]
y_train = y_train[:5000]

reset_graph()

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int32, shape=(None), name="y")

with tf.name_scope("dnn"):
    hidden1=tf.layers.dense(X, n_hidden1,name="hidden1",activation=tf.nn.relu)
    hidden2=tf.layers.dense(hidden1, n_hidden2,name="hidden2",activation=tf.nn.relu)
    logits=tf.layers.dense(hidden2, n_outputs,name="outputs",activation=None)
with tf.name_scope("loss"):
    xentropy=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=logits)
    loss=tf.reduce_mean(xentropy,name="loss")
    loss_summary=tf.summary.scalar("log_loss", loss)
with tf.name_scope("train"):
    optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    training_op=optimizer.minimize(loss)
with tf.name_scope("evalue"):
    correct=tf.nn.in_top_k(logits, y, 1)
    accuracy=tf.reduce_mean(tf.cast(correct,tf.float32))
    accuracy_summary=tf.summary.scalar("accuracy", accuracy)
with tf.name_scope("init"):
    init=tf.global_variables_initializer()
with tf.name_scope("save"):
    saver=tf.train.Saver()


from datetime import datetime

def log_dir(prefix=""):
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "tf_logs"
    if prefix:
        prefix += "-"
    name = prefix + "run-" + now   return "{}/{}/".format(root_logdir, name)

logdir = log_dir("mnist_dnn")
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
n_epochs=20
batch_size=50
n_batches=int(X_train.shape[0]/batch_size)
checkpoint_path="/tmp/my_deep_mnist_model.ckpt"
checkpoint_epoch_path=checkpoint_path+".epoch"
final_model_path="./my_deep_mnist_model"

best_loss=np.infty
epochs_without_process=0
max_epochs_without_process=50

with tf.Session() as sess:
    if os.path.isfile(checkpoint_epoch_path):
        with open(checkpoint_epoch_path,"rb") as f:
            start_epoch=int(f.read())
        saver.restore(sess, checkpoint_path)
    else:
        start_epoch=0
        init.run()
    for epoch in range(start_epoch,n_epochs):
        for X_batch,y_batch in shuffle_batch(X_train, y_train, batch_size):
            sess.run(training_op,feed_dict={X:X_batch,y:y_batch})
        accuracy_val, loss_val, accuracy_summary_str, loss_summary_str = sess.run([accuracy, loss, accuracy_summary, loss_summary],feed_dict={X:X_batch,y:y_batch})
        file_writer.add_summary(accuracy_summary_str,epoch)
        file_writer.add_summary(loss_summary_str,epoch)
        if epoch % 5 ==0:
            print("accuracy :",accuracy_val,"loss :",loss_val)
            saver.save(sess, checkpoint_path)
            with open(checkpoint_epoch_path,"wb") as f:
                f.write(b"%d" %(epoch+1))
            if loss_val<best_loss:
                saver.save(sess, final_model_path)
                best_loss=loss_val
            else:
                epochs_without_process=+5
                if epochs_without_process>max_epochs_without_process:
                    print("Early stopping")
                    break

