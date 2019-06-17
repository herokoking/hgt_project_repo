#!/usr/bin/python3
# this script is used to try learning_rate_decay
#!/usr/bin/python3
# github 已经建立好的cnn 神经网络（包括：alexnet，inceptionnet，resnet等）
# 试用cifar net （针对cifar10建立的）
import tensorflow as tf
import pandas as pd
import numpy as np
import cifarnet
import tensorflow.contrib.slim as slim
# 定义learning_rate随着epoch改变而改变，当循环epoch<50次时候，learning_rate=0.1,当epoch超过50次后，learning_rate=0.01，从而实现learning_rate衰减，前期快速降低，后期缓降，从而在短时间内到达loss最小值


def lr_step(step, **kwargs):
    lr = tf.cond(tf.less(step, 50), lambda: 0.1, lambda: 0.01)
    return lr


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
y_train = y_train.reshape([-1, ])
y_test = y_test.astype(np.int32)
y_test = y_test.reshape([-1, ])
X_valid, X_train = X_train[:50], X_train[50:300]
y_valid, y_train = y_train[:50], y_train[50:300]
X_test = X_test[:50]
y_test = y_test[:50]

reset_graph()
X = tf.placeholder(tf.float32, shape=[None, 32, 32, 3], name="X")
y = tf.placeholder(tf.int32, shape=[None], name="y")
step = tf.placeholder(tf.float32, name='step')
n_epochs = 50
batch_size = 20
max_checks_without_progress = 20
checks_without_progress = 0
best_loss = np.infty

with tf.name_scope("resnet"):
    with slim.arg_scope(cifarnet.cifarnet_arg_scope()):
        logits, end_points = cifarnet.cifarnet(X)
with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=y)
    loss = tf.reduce_mean(xentropy)
with tf.name_scope("train"):
    change_lr = lr_step(step)
    optimizer = tf.train.MomentumOptimizer(
        learning_rate=change_lr, momentum=0.9)
    training_op = optimizer.minimize(loss)
with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
init = tf.global_variables_initializer()
saver = tf.train.Saver()
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            sess.run(training_op, feed_dict={
                     X: X_batch, y: y_batch, step: epoch})
        loss_value = loss.eval(feed_dict={X: X_valid, y: y_valid, step: epoch})
        if loss_value < best_loss:
            best_loss = loss_value
            save_path = saver.save(sess, "./cifarnet_final_model.ckpt")
            checks_without_progress = 0
        else:
            checks_without_progress += 1
            if checks_without_progress > max_checks_without_progress:
                print("early stopping")
                break
        accuracy_value = accuracy.eval(
            feed_dict={X: X_valid, y: y_valid, step: epoch})
        print("accuracy :", accuracy_value, "loss_value is : ", loss_value)
with tf.Session() as sess:
    saver.restore(sess, "./cifarnet_final_model.ckpt")
    test_acc_val = accuracy.eval(feed_dict={X: X_test, y: y_test})
    print("test_dataset accuracy is :", test_acc_val)
