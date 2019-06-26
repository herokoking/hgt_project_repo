#!/usr/bin/python3
#this script is for chapter11 exercise 8,9,10
#example in the book
import numpy as np
import pandas as pd
import os
import tensorflow as tf
import matplotlib.pyplot as plt

%matplotlib inline
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)
plt.rcParams['labelsize']=14
plt.rcParams['xtick.labelsize']=12
plt.rcParams['ytick.labelsize']=12

def save_fig(fig_id,tight_layout=True):
    path=os.path.join("./imagesdeep"+fig_id+".png")
    print("saving figure",fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path,format='png',dpi=300)

def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch


n_inputs=28*28
n_hidden1=300
n_hidden2=200
n_hidden3=100
batch_size=50
learning_rate=0.01
n_epochs=100

##解决梯度消失/爆炸的方法

#Method1 权重随机初始化
reset_graph()

X=tf.placeholder(tf.float32,shape=(None,n_inputs),name="X")
he_init=tf.variance_scaling_initializer()
hidden1=tf.layers.dense(X, n_hidden1,activation=tf.nn.relu,kernel_initializer=he_init,name="hidden1")

#Method2 更换不饱和激活函数，用ELU（加速线性单元）或Leaky ReLU（代替ReLU（修正线性单元）
reset_graph()
X=tf.placeholder(tf.float32,shape=(None,n_inputs),name="X")
def leaky_relu(z,name=None):
    return tf.maximum(0.01*z, z,name=name)
hidden1=tf.layers.dense(X, n_hidden1,activation=leaky_relu,name="hidden1")
#或者选择ELU作为激活函数
hidden1=tf.layers.dense(X, n_hidden1,activation=tf.nn.elu,name="hidden1")
#又或者使用SELU（不能正则化）
hidden1=tf.layers.dense(X, n_hidden1,activation=tf.nn.selu,name="hidden1")

#Method3 批量标准化(Batch Normalization)
#注意使用新功能tf.layers.batch_normalization()，需要再定义一个节点extra_update_ops，后面的session中也需run此extra_update_ops节点
reset_graph()

X=tf.placeholder(tf.float32,shape=(None,n_inputs),name="X")
y=tf.placeholder(tf.int32,shape=(None),name="y")
training=tf.placeholder_with_default(False, shape=(),name="training")
with tf.name_scope("dnn"):
    hidden1=tf.layers.dense(X, n_hidden1,name="hidden1")
    batch_normalization1=tf.layers.batch_normalization(hidden1,training=training,momentum=0.9)
    bn_hidden1=tf.nn.elu(batch_normalization1)
    hidden2=tf.layers.dense(bn_hidden1, n_hidden2,name='hidden2')
    batch_normalization2=tf.layers.batch_normalization(hidden2,training=training,momentum=0.9)
    bn_hidden2=tf.nn.elu(batch_normalization2)
    hidden3=tf.layers.dense(bn_hidden2, n_hidden3,name='hidden2')
    batch_normalization3=tf.layers.batch_normalization(hidden3,training=training,momentum=0.9)
    bn_hidden3=tf.nn.elu(batch_normalization3)
    logits_before_bn=tf.layers.dense(bn_hidden3, n_outputs,name='outputs')
    logits=tf.layers.batch_normalization(logits_before_bn,training=training,momentum=0.9)
    extra_update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.name_scope("loss"):
    xentropy=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=logits)
    loss=tf.reduce_mean(xentropy,name="loss")
with tf.name_scope("train"):
    optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    training_op=optimizer.minimize(loss)
with tf.name_scope("eval"):
    correct=tf.nn.in_top_k(logits, y, 1)
    accuracy=tf.reduce_mean(tf.cast(correct,tf.float32))
with tf.name_scope("init"):
    init=tf.global_variables_initializer()
with tf.name('save'):
    saver=tf.train.Saver()

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for X_batch,y_batch in shuffle_batch(X_train,y_train,batch_size):
            sess.run([training_op,extra_update_ops],feed_dict={X:X_batch,y:y_batch,training:True})
        accuracy_val=accuracy.eval(feed_dict={X:X_batch,y:y_batch,training:True})
        print(accuracy_val)

#Method4 梯度裁剪 运用于定义train域中,自定义的梯度裁剪规则替代传统的minimize
with tf.name_scope("train"):
    optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    grads_and_vars=optimizer.compute_gradients(loss)
    capped_gvs=[(tf.clip_by_value(grad,-threshold,threshold),var) for grad,var in grads_and_vars]
    training_op=optimizer.apply_gradients(capped_gvs)





##迁移学习（重用预训练图层） reuse pretrained layers

#基础部分 重用模型
reset_graph()
#import_meta_graph()函数导入图的结构和所有操作到default graph上，返回一个saver用来后面恢复模型的状态
saver=tf.train.import_meta_graph("./my_model_final.ckpt.meta")   
with tf.Session() as sess:
    saver.restore(sess, "./my_model_final.ckpt")
    for epoch in range(n_epochs):
        for X_batch,y_batch in shuffle_batch(X_train,y_train,batch_size):
            sess.run(training_op,feed_dict={X:X_batch,y:y_batch,training:True})
        accuracy_val=accuracy.eval(feed_dict={X:X_batch,y:y_batch,training:True})
        print(accuracy_val)

#只使用原来model的前几层，然后自己构建后面的隐藏层和输出层
#for example reuse pretrained 2nd layers , build new 3rd layer(没冻结任何层)
reset_graph()
n_hidden3=50
n_outputs=10
saver=tf.train.import_meta_graph("./my_model_final.ckpt.meta")
X=tf.get_default_graph().get_tensor_by_name("X:0")
y=tf.get_default_graph().get_tensor_by_name("y:0")
with tf.name_scope('new_dnn'):
    hidden2=tf.get_default_graph().get_tensor_by_name("dnn/hidden2/ReLU:0")
    new_hidden3=tf.layers.dense(hidden2, n_hidden3,activation=tf.nn.elu,name="new_hidden3")
    new_logits=tf.layers.dense(new_hidden3, n_outputs,name='new_outputs')

with tf.name_scope("new_loss"):
    xentropy=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=new_logits)
    loss=tf.reduce_mean(xentropy,name='loss')
with tf.name_scope('new_train'):
    optimizer=tf.train.GradientDescentOptimizer(learning_rate)
    training_op=optimizer.minimize(loss)
init = tf.global_variables_initializer()
new_saver = tf.train.Saver()

with tf.Session() as sess:
    init.run()
    saver.restore(sess, "./my_model_final.ckpt")

    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        accuracy_val = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
        print(epoch, "Validation accuracy:", accuracy_val)

    save_path = new_saver.save(sess, "./my_new_model_final.ckpt")

'''
#获得默认计算图中所有节点
tf.get_default_graph().get_operations()
#get a handle on the variables using get_collection() and specifying the scope（获得指定域的变量句柄）
tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="hidden1")            #获得scope：hidden1的所有变量
tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="new_outputs")            #获得scope：new_outputs的所有训练变量
#另一种方法 使用get_tensor_by_name()
tf.get_default_graph().get_tensor_by_name("hidden1/kernel:0")
tf.get_default_graph().get_tensor_by_name("hidden1/bias:0")
'''

#冻结低层(两种方法)
#Method1 在"train"的scope 中指定重新训练的变量列表
reset_graph()
n_inputs=28*28
n_hidden1=30
n_hidden2=20
n_hidden3=10
n_outputs=10

X=tf.placeholder(tf.float32,shape=(None,n_inputs),name="X")
y=tf.placeholder(tf.int32,shape=(None),name="y")

with tf.name_scope("dnn"):
    hidden1=tf.layers.dense(X, n_hidden1,activation=tf.nn.relu,name="hidden1")
    hidden2=tf.layers.dense(hidden1, n_hidden2,activation=tf.nn.relu,name="hidden2")
    hidden3=tf.layers.dense(hidden2, n_hidden3,activation=tf.nn.relu,name="hidden3")
    logits=tf.layers.dense(hidden3, n_outputs,name="outputs")
with tf.name_scope("loss"):
    xentropy=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=logits)
    loss=tf.reduce_mean(xentropy,name="loss")
with tf.name_scope("eval"):
    correct=tf.nn.in_top_k(logits, y, 1)
    accuracy=tf.reduce_mean(tf.cast(correct,tf.float32),name="accuracy")
with tf.name_scope("train"):
    optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_vars=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="hidden[3]|outputs")    #hidden[3]|outputs 这是正则匹配，收集要训练的变量
    training_op=optimizer.minimize(loss,var_list=train_vars)                                    #重新训练


#reuse hidden1 and hidden2 layers
reuse_vars=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="hidden[12]")                  #hidden[12]这是一个正则匹配, 收集要重用的变量
#restore hidden1 layer and hidden2 layer
restore_saver=tf.train.Saver(reuse_vars)

init=tf.global_variables_initializer()
new_saver=tf.train.Saver()

with tf.Session() as sess:
    init.run()
    restore_saver.restore(sess, "./my_model_final.ckpt")
    for epoch in range(n_epochs):
        for X_batch,y_batch in shuffle_batch(X_train, y_train, batch_size):
            sess.run(training_op,feed_dict={X:X_batch,y:y_batch})
        accuracy_val=accuracy.eval(feed_dict={X:X_batch,y:y_batch})
        print(accuracy_val)
    save_path=new_saver.save(sess, "./my_new_model_final.ckpt")

#Method2 在"dnn"的scope中通过tf.stop_gradient()函数指定hidden1层不参与梯度下降更新权重的过程
reset_graph()
n_inputs=28*28
n_hidden1=30
n_hidden2=20
n_hidden3=10
n_outputs=10

X=tf.placeholder(tf.float32,shape=(None,n_inputs),name="X")
y=tf.placeholder(tf.int32,shape=(None),name="y")

with tf.name_scope("dnn"):
    hidden1=tf.layers.dense(X, n_hidden1,activation=tf.nn.relu,name="hidden1")              # reused frozen
    hidden1_stop=tf.stop_gradient(hidden1)                                                  
    hidden2=tf.layers.dense(hidden1_stop, n_hidden2,activation=tf.nn.relu,name="hidden2")        # reused, not frozen
    hidden3=tf.layers.dense(hidden2, n_hidden3,activation=tf.nn.relu,name="hidden3")
    logits=tf.layers.dense(hidden3, n_outputs,name="outputs")
with tf.name_scope("loss"):
    xentropy=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=logits)
    loss=tf.reduce_mean(xentropy,name="loss")
with tf.name_scope("eval"):
    correct=tf.nn.in_top_k(logits, y, 1)
    accuracy=tf.reduce_mean(tf.cast(correct,tf.float32),name="accuracy")
with tf.name_scope("train"):
    optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    training_op=optimizer.minimize(loss)

#reuse hidden1 and hidden2 layers
reuse_vars=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="hidden[12]")                  #hidden[12]这是一个正则匹配, 收集要重用的变量
#restore hidden1 layer and hidden2 layer
restore_saver=tf.train.Saver(reuse_vars)

init=tf.global_variables_initializer()
new_saver=tf.train.Saver()

with tf.Session() as sess:
    init.run()
    restore_saver.restore(sess, "./my_model_final.ckpt")
    for epoch in range(n_epochs):
        for X_batch,y_batch in shuffle_batch(X_train, y_train, batch_size):
            sess.run(training_op,feed_dict={X:X_batch,y:y_batch})
        accuracy_val=accuracy.eval(feed_dict={X:X_batch,y:y_batch})
        print(accuracy_val)
    save_path=new_saver.save(sess, "./my_new_model_final.ckpt")



##缓存冻结的低层(caching the frozen layers)
#冻结的层不会改变，所以对冻结层的最上层用一个训练样本做缓存，这样做每个训练实例在冻结层中执行一次，而不是每一个epoch都会被执行
reset_graph()
n_inputs=28*28
n_hidden1=300
n_hidden2=50
n_hidden3=20
n_outputs=10

X=tf.placeholder(tf.float32,shape=(None,n_inputs),name="X")
y=tf.placeholder(tf.int32,shape=(None),name="y")

with tf.name_scope("dnn"):
    hidden1=tf.layers.dense(X, n_hidden1,activation=tf.nn.relu,name="hidden1")              # reused frozen & cached
    hidden1_stop=tf.stop_gradient(hidden1)                                                  
    hidden2=tf.layers.dense(hidden1_stop, n_hidden2,activation=tf.nn.relu,name="hidden2")        # reused, not frozen
    hidden3=tf.layers.dense(hidden2, n_hidden3,activation=tf.nn.relu,name="hidden3")
    logits=tf.layers.dense(hidden3, n_outputs,name="outputs")
with tf.name_scope("loss"):
    xentropy=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=logits)
    loss=tf.reduce_mean(xentropy,name="loss")
with tf.name_scope("eval"):
    correct=tf.nn.in_top_k(logits, y, 1)
    accuracy=tf.reduce_mean(tf.cast(correct,tf.float32),name="accuracy")
with tf.name_scope("train"):
    optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    training_op=optimizer.minimize(loss)

#reuse hidden1 and hidden2 layers
reuse_vars=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="hidden[12]")                  #hidden[12]这是一个正则匹配, 收集要重用的变量
#restore hidden1 layer and hidden2 layer
restore_saver=tf.train.Saver(reuse_vars)

init=tf.global_variables_initializer()
new_saver=tf.train.Saver()

with tf.Session() as sess:
    init.run()
    restore_saver.restore(sess, "./my_model_final.ckpt")

    h1_cache=sess.run(hidden1,feed_dict={X:X_train})
    h1_cache_valid=sess.run(hidden1,feed_dict={X:X_valid})

    for epoch in range(n_epochs):
        #把hidden1层转换为批量输入
        shuffled_idx = np.random.permutation(len(X_train))
        hidden1_batches=np.array_split(h1_cache[shuffled_idx],n_batches)
        y_batches=np.array_split(y_train[shuffled_idx], n_batches)

        for hidden1_batch,y_batch in zip(hidden1_batches, y_batches):
            sess.run(training_op,feed_dict={hidden1:hidden1_batch,y:y_batch})       #批量代入换成了hidden1_batch
        accuracy_val=accuracy.eval(feed_dict={hidden1:hidden1_batch,y:y_batch})
        print(accuracy_val)
    save_path=new_saver.save(sess, "./my_new_model_final.ckpt")


##更换优化器（faster optimizers） Momentum optimization, Nesterov Accelerated Gradient, AdaGrad, RMSProp, Adam Optimization
#常用Adam Optimization

##学习速率调度(learning rate scheduling)----定义"train" scope,让learning_rate指数变化而不是定值
#常用指数调度
with tf.name_scope("train"):
    initial_learning_rate=0.1
    decay_steps=1000
    decay_rate=1/10
    global_step=tf.Variable(0,trainable=False,name="global_step")
    learning_rate=tf.train.exponential_decay(initial_learning_rate, global_step, decay_steps, decay_rate)
    optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    training_op=optimizer.minimize(loss)

##通过正则化避免过拟合
#Method1 l1正则化连接权重向量
reset_graph()
n_inputs=28*28
n_hidden1=30
n_hidden2=20
n_outputs=10

X=tf.placeholder(tf.float32,shape=(None,n_inputs),name="X")
y=tf.placeholder(tf.int32,shape=(None),name="y")

with tf.name_scope("dnn"):
    hidden1=tf.layers.dense(X, n_hidden1,activation=tf.nn.relu,name="hidden1")
    hidden2=tf.layers.dense(hidden1, n_hidden2,activation=tf.nn.relu,name="hidden2")
    hidden3=tf.layers.dense(hidden2, n_hidden3,activation=tf.nn.relu,name="hidden3")
    logits=tf.layers.dense(hidden3, n_outputs,name="outputs")
#取出四个层的权重
W1=tf.get_default_graph().get_tensor_by_name("hidden1/kernel:0")
W2=tf.get_default_graph().get_tensor_by_name("hidden2/kernel:0")
W3=tf.get_default_graph().get_tensor_by_name("hidden3/kernel:0")
W4=tf.get_default_graph().get_tensor_by_name("outputs/kernel:0")
scale=0.001
#把四层的权重的正则化损失+交叉熵作为loss值
with tf.name_scope("loss"):
    xentropy=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=logits)
    base_loss=tf.reduce_mean(xentropy,name="avg_loss")
    reg_losses=tf.reduce_sum(tf.abs(W1))+tf.reduce_sum(tf.abs(W2))+tf.reduce_sum(tf.abs(W3))+tf.reduce_sum(tf.abs(W4))
    loss=tf.add(base_loss, scale*reg_losses,name="loss")
with tf.name_scope("eval"):
    correct=tf.nn.in_top_k(logits, y, 1)
    accuracy=tf.reduce_mean(tf.cast(correct,tf.float32),name="accuracy")
with tf.name_scope("train"):
    optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    training_op=optimizer.minimize(loss)

#Method2 l1正则化连接权重向量（另一种形式）
#利用partial()把正则化加入到tf.layers.dense()中
reset_graph()
n_inputs=28*28
n_hidden1=30
n_hidden2=20
n_outputs=10

X=tf.placeholder(tf.float32,shape=(None,n_inputs),name="X")
y=tf.placeholder(tf.int32,shape=(None),name="y")

scale=0.001
my_dense_layer=partial(tf.layers.dense,activation=tf.nn.relu,kernel_regularizer=tf.contrib.layers.l1_regularizer(scale))
with tf.name_scope("dnn"):
    hidden1=my_dense_layer(X,n_hidden1,name="hidden1")
    hidden2=my_dense_layer(hidden1,n_hidden2,name="hidden2")
    logits=my_dense_layer(hidden2,n_outputs,activation=None,name="outputs")
with tf.name_scope("loss"):
    xentropy=tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    base_loss=tf.reduce_mean(xentropy,name="avg_loss")
    reg_losses=tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss=tf.add(base_loss, scale*reg_losses,name="loss")

with tf.name_scope("eval"):
    correct=tf.nn.in_top_k(logits, y, 1)
    accuracy=tf.reduce_mean(tf.cast(correct,tf.float32),name="accuracy")
with tf.name_scope("train"):
    optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    training_op=optimizer.minimize(loss)

#Method3 Dropout
reset_graph()
n_inputs=28*28
n_hidden1=30
n_hidden2=20
n_outputs=10

X=tf.placeholder(tf.float32,shape=(None,n_inputs),name="X")
y=tf.placeholder(tf.int32,shape=(None),name="y")
training=tf.placeholder_with_default(False, shape=(),name="training")
dropout_rate=0.5
X_drop=tf.layers.dropout(X,dropout_rate,training=training)
with tf.name_scope("dnn"):
    hidden1=tf.layers.dense(X_drop, n_hidden1,activation=tf.nn.relu,name="hidden1")
    hidden1_drop=tf.layers.dropout(hidden1,dropout_rate,training=training)
    hidden2=tf.layers.dense(hidden1_drop, n_hidden2,activation=tf.nn.relu,name="hidden2")
    hidden2_drop=tf.layers.dropout(hidden2,dropout_rate,training=training)
    logits=tf.layers.dense(hidden2_drop,n_outputs,name="outputs")
with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)    

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    
init = tf.global_variables_initializer()
saver = tf.train.Saver()

n_epochs = 20
batch_size = 50

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch, training: True})
        accuracy_val = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
        print(epoch, "Validation accuracy:", accuracy_val)

    save_path = saver.save(sess, "./my_model_final.ckpt")


#Method4 Max norm(最大范数正则化)
reset_graph()
n_inputs=28*28
n_hidden1=30
n_hidden2=20
n_outputs=10
X=tf.placeholder(tf.float32,shape=(None,n_inputs),name="X")
y=tf.placeholder(tf.int32,shape=(None),name="y")

with tf.name_scope("dnn"):
    hidden1=tf.layers.dense(X, n_hidden1,activation=tf.nn.relu,name="hidden1")
    hidden2=tf.layers.dense(hidden1, n_hidden2,activation=tf.nn.relu,name="hidden2")
    logits=tf.layers.dense(hidden2, n_outputs,name="outputs")
with tf.name_scope("loss"):
    xentropy=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=logits)
    loss=tf.reduce_mean(xentropy,name="loss")
with tf.name_scope("train"):
    optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    training_op=optimizer.minimize(loss)
with tf.name_scope("eval"):
    correct=tf.nn.in_top_k(logits,y, 1)
    accuracy=tf.reduce_mean(tf.cast(correct, tf.float32))

threshold=1.0
weights=tf.get_default_graph().get_tensor_by_name("hidden1/kernel:0")
clipped_weights=tf.clip_by_norm(weights, clip_norm=threshold,axes=1)
clip_weights=tf.assign(weights, clipped_weights)
weights2=tf.get_default_graph().get_tensor_by_name("hidden2/kernel:1")
clipped_weights2=tf.clip_by_norm(weights2, clip_norm=threshold,axes=1)
clip_weights2=tf.assign(weights2, clipped_weights2)

init=tf.global_variables_initializer()
saver=tf.train.Saver()

with tf.Session() as sess:
    init.run()
    for epoch in n_epochs:
        for X_batch,y_batch in shuffle_batch(X_train,y_train,batch_size):
            training_op.run(feed_dict={X:X_batch,y:y_batch})
            sess.run([clip_weights,clip_weights2])                          #在运行training_op节点后，运行对权重的最大范数正则化，迭代更新权重
        accuracy_val=accuracy.run(feed_dict={X:X_batch,y:y_batch})
    save_path=saver.save(sess, "./my_model_final.ckpt")

#替代方法，先定义一个最大范数正则化的函数
def max_norm_regularizer(threshold,axex=1,name="max_norm",collection="max_norm"):
    def max_norm(weights):
        clipped=tf.clip_by_norm(weights, clip_norm=threshold,axes=axes)
        clip_weights=tf.assign(weights, clipped)
        tf.add_to_collection(collection, clip_weights)
        return None
    return max_norm

reset_graph()
n_inputs=28*28
n_hidden1=30
n_hidden2=20
n_outputs=10
X=tf.placeholder(tf.float32,shape=(None,n_inputs),name="X")
y=tf.placeholder(tf.int32,shape=(None),name="y")
max_norm_reg = max_norm_regularizer(threshold=1.0)
with tf.name_scope("dnn"):
    hidden1=tf.layers.dense(X, n_hidden1,activation=tf.nn.relu,kernel_regularizer=max_norm_reg,name="hidden1")
    hidden2=tf.layers.dense(hidden1, n_hidden2,activation=tf.nn.relu,kernel_regularizer=max_norm_reg,name="hidden2")
    logits=tf.layers.dense(hidden2, n_outputs,name="outputs")
with tf.name_scope("loss"):
    xentropy=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=logits)
    loss=tf.reduce_mean(xentropy,name="loss")
with tf.name_scope("train"):
    optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    training_op=optimizer.minimize(loss)
with tf.name_scope("eval"):
    correct=tf.nn.in_top_k(logits,y, 1)
    accuracy=tf.reduce_mean(tf.cast(correct, tf.float32))

init=tf.global_variables_initializer()
saver=tf.train.Saver()

clip_all_weights = tf.get_collection("max_norm")

with tf.Session() as sess:
    init.run()
    for epoch in n_epochs:
        for X_batch,y_batch in shuffle_batch(X_train,y_train,batch_size):
            training_op.run(feed_dict={X:X_batch,y:y_batch})
            sess.run(clip_all_weights)                          #在运行training_op节点后，运行对权重的最大范数正则化，迭代更新权重
        accuracy_val=accuracy.run(feed_dict={X:X_batch,y:y_batch})
    save_path=saver.save(sess, "./my_model_final.ckpt")


##exercise8
#8.1 define a dnn layer
'''
with tf.name_scope("dnn"):
    he_init=tf.variance_scaling_initializer()
    hidden1=tf.layers.dense(X, 100,activation=tf.nn.elu,kernel_initializer=he_init,name="hidden1")
    hidden2=tf.layers.dense(hidden1, 100,activation=tf.nn.elu,kernel_initializer=he_init,name="hidden2")
    hidden3=tf.layers.dense(hidden2, 100,activation=tf.nn.elu,kernel_initializer=he_init,name="hidden3")
    hidden4=tf.layers.dense(hidden3, 100,activation=tf.nn.elu,kernel_initializer=he_init,name="hidden4")
    hidden5=tf.layers.dense(hidden4, 100,activation=tf.nn.elu,kernel_initializer=he_init,name="hidden5")
    logits=tf.layers.dense(hidden5, n_outputs,name="outputs")
'''
he_init=tf.variance_scaling_initializer()
def dnn(inputs,n_hidden_layers=5,n_neurons=100,name=None,activation=tf.nn.elu,initializer=he_init):
    with tf.name_scope("dnn"):
        for layer in range(n_hidden_layers):
            inputs=tf.layers.dense(inputs, n_neurons,activation=activation,kernel_initializer=initializer,name="hidden%d" % (layer+1))
        return inputs
dnn_outputs=dnn(X)
logits=tf.layers.dense(dnn_outputs, n_outputs,kernel_initializer=he_init,name="outputs")
y_prob=tf.nn.softmax(logits,name="Y_proba")

#8.2
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype(np.float32).reshape(-1, 28*28) / 255.0
X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
X_valid, X_train = X_train[:5000], X_train[5000:]
y_valid, y_train = y_train[:5000], y_train[5000:]

X_train1=X_train[y_train<5]
y_train1=y_train[y_train<5]
X_valid1=X_valid[y_valid<5]
y_valid1=y_valid[y_valid<5]
X_test1=X_test[y_test<5]
y_test1=y_test[y_test<5]


reset_graph()
n_inputs=28*28
n_hidden1=100
n_hidden2=50
n_outputs=5
X=tf.placeholder(tf.float32,shape=(None,n_inputs),name="X")
y=tf.placeholder(tf.int32,shape=(None),name="y")
learning_rate=0.01

with tf.name_scope("dnn"):
    hidden1=tf.layers.dense(X, n_hidden1,activation=tf.nn.relu,name="hidden1")
    hidden2=tf.layers.dense(hidden1, n_hidden2,activation=tf.nn.relu,name="hidden2")
    logits=tf.layers.dense(hidden2, n_outputs,name="outputs")
with tf.name_scope("loss"):
    xentropy=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=logits)
    loss=tf.reduce_mean(xentropy,name="loss")
with tf.name_scope("train"):
    optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op=optimizer.minimize(loss)
with tf.name_scope("eval"):
    correct=tf.nn.in_top_k(logits,y, 1)
    accuracy=tf.reduce_mean(tf.cast(correct, tf.float32))

n_epochs = 100
batch_size = 50

max_checks_without_progress = 20
checks_without_progress = 0
best_loss = np.infty

init=tf.global_variables_initializer()
saver=tf.train.Saver()
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for X_batch,y_batch in shuffle_batch(X_train1, y_train1, batch_size):
            training_op.run(feed_dict={X:X_batch,y:y_batch})
        loss_value=loss.eval(feed_dict={X:X_valid1,y:y_valid1})
        if loss_value < best_loss:
            best_loss=loss_value
            save_path=saver.save(sess, "./my_model_final.ckpt")
            checks_without_progress=0
        else:
            checks_without_progress+=1
            if checks_without_progress>max_checks_without_progress:
                print("early stopping")
                break
        accuracy_value=accuracy.eval(feed_dict={X:X_valid1,y:y_valid1})
        print("accuracy :",accuracy_value,"loss_value is : ",loss_val)
with tf.Session() as sess:
    saver.restore(sess, "./my_model_final.ckpt")
    accuracy_testdata=accuracy.eval(feed_dict={X:X_test1,y:y_test1})
    print("Final test accuracy: {:.2f}%".format(accuracy_testdata*100))

#8.3
from sklearn.base import BaseEstimator,ClassifierMixin
from sklearn.exceptions import NotFittedError
he_init=tf.variance_scaling_initializer()

class DNNClassifier (BaseEstimator,ClassifierMixin):
    def __init__ (
        self,n_hidden_layers=5,n_neurons=100,optimizer_class=tf.train.AdamOptimizer,
        learning_rate=0.01,batch_size=20,activation=tf.nn.elu,initializer=he.init,
        batch_norm_momentum=None,dropout_rate=None,random_state=None):
    self.n_hidden_layers=n_hidden_layers
    self.n_neurons=n_neurons
    self.optimizer_class=optimizer_class
    self.learning_rate=learning_rate
    self.batch_size=batch_size
    self.activation=activation
    self.initializer=initializer
    self.batch_norm_momentum=batch_norm_momentum
    self.dropout_rate=dropout_rate
    self.random_state=random_state
    self._session=None
    def _dnn(self,inputs):
        for layer in range(self.n_hidden_layers):
            if self.dropout_rate:
                inputs=tf.layers.dropout(inputs,self.dropout_rate,training=self._training)
            inputs=tf.layers.dense(inputs, self.n_neurons,kernel_initializer=self.initializer,name="hidden_layer%d" %(layer+1))
            if self.batch_norm_momentum:
                inputs=tf.layers.batch_normalization(inputs,momentum=self.batch_norm_momentum,training=self._training)
            inputs=self.activation(inputs, name="hidden%d_out"%(layer+1))
        return inputs
    def _build_graph(self,n_inputs,n_outputs):
        if self.random_state is not None:
            tf.set_random_seed(self.random_state)
            np.random.seed(self.random_state)
        X=tf.placeholder(tf.float32,shape=(None,n_inputs),name="X")
        y=tf.placeholder(tf.int32,shape=(None),name="y")
        if self.batch_norm_momentum or self.dropout_rate:
            self._training=tf.placeholder_with_default(False,shape=(),name="training")
        else:
            self.training=None
        dnn_outputs=self._dnn(X)
        logits=tf.layers.dense(dnn_output, n_outputs,kernel_initializer=self.initializer,name="outputs")
        Y_proba=tf.nn.softmax(logits,name="y_proba")
        xentropy=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=logits,name="xentropy")
        loss=tf.reduce_mean(xentropy,name="loss")
        optimizer=tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        training_op=optimizer.minimize(loss)
        correct=in_top_k(logits,y,1)
        accuracy=tf.reduce_mean(tf.cast(correct,tf.float32),name="accuracy")

        init=tf.global_variables_initializer()
        saver=tf.train.Saver()
        
        self._X, self._y = X, y
        self._Y_proba, self._loss = Y_proba, loss
        self._training_op, self._accuracy = training_op, accuracy
        self._init, self._saver = init, saver

    def close_session(self):
        if self._session:
            self._session.close()
    def _get_model_params(self):
        with self._graph.as_defult():
            gvars=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        return {gvars.op.name:value for gvars,value in zip (gvars,self._session.run(gvars))}
    def _restore_model_params(self,model_params):
        gvar_names=list(model_params.keys())
        assign_ops={gvar_name:self._graph.get_operation_by_name(gvar_name+"/Assign") for gvar_name in gvar_names}
        init_values={gvar_name:assign_op.inputs[1] for gvar_name,assign_op in assign_ops.items()}
        feed_dict={init_values[gvar_name]:model_params[gvar_name] for gvar_name in gvar_names}
        self._session.run(assign_ops,feed_dict=feed_dict)
    def fit(self,X,y,n_epochs=100,X_valid=None,y_valid=None):
        self.close_session()
        n_inputs=X.shape[1]
        self.classes_=np.unique(y)
        n_outputs=len(self.classes_)
        self.class_to_index_={label:index for index,label in enumerate(self.classes_)}
        y=np.array([self.class_to_index_[label] for label in y], dtype=np.int32)
        self._graph = tf.Graph()
        with self._graph.as_default():
            self._build_graph(n_inputs, n_outputs)
            extra_update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        max_checks_without_progress=20
        checks_without_progress=0
        best_loss=np.infty
        best_params=None
        self._session=tf.Session(graph=self._graph)
        with self._session.as_default() as sess:
            self._init.run()
            for epoch in range(n_epochs):
                for X_batch,y_batch in shuffle_batch(X_train, y_train, batch_size)
                    if self._training is not None:
                        feed_dict[self._training]=True
                    sess.run(self._training_op,feed_dict=feed_dict)
                    if extra_update_ops:
                        sess.run(extra_update_ops,feed_dict=feed_dict)
                if X_valid is not None and y_valid is not None:
                    loss_value,accuracy_val=sess.run([loss,accuracy],feed_dict={X:X_val,y:y_val})
                    if loss_value<best_loss:
                        best_loss=loss_value
                        best_params = self._get_model_params()
                        checks_without_progress=0
                    else:
                        checks_without_progress+=1
                        if checks_without_progress>max_checks_without_progress:
                            print("early stop")
                            break
                else:
                    loss_train,acc_train=sess.run([self._loss,self._accuracy],feed_dict={X:X_batch,y:y_batch})
            if best_params:
                self._restore_model_params(best_params)
            return self
    def predict_proba(self,X):
        if not self._session:
            raise NotFittedError("this %s instance is not fitted yet" %self.__class__.__name__)
            with self.__Y_proba.eval(feed_dict={self._X:X})
    def predict(self,X):
        class_indices=np.argmax(self.predict_proba(X),axis=1)
        return np.array([self.class_[class_index]] for class_index in class_indices,np.int32)
    def save(self,path):
        self._saver.save(self._session, path)

#run
dnn_clf=DNNClassifier(random_state=42)
dnn_clf.fit(X_train, y_train,n_epochs=1000,X_valid=X_valid1,y_valid=y_valid1)
from sklearn.metrics import accuracy_score
y_pred=dnn_clf.predict(X_test1)
print(accuracy_score(y_test1, y_pred))

#run CrossValidation to search best hyperparameters
from sklearn.model_selection import RandomizedSearchCV

param_distribs = {
    "n_neurons": [10, 30, 50, 70, 90, 100, 120, 140, 160],
    "batch_size": [10, 50, 100, 500],
    "learning_rate": [0.01, 0.02, 0.05, 0.1],
    "activation": [tf.nn.relu, tf.nn.elu],
    "optimizer_class":[tf.train.AdamOptimizer,tf.train.GradientDescentOptimizer]
    }
rnd_search = RandomizedSearchCV(DNNClassifier(random_state=42), param_distribs, n_iter=50,cv=3, random_state=42, verbose=2)
rnd_search.fit(X_train1, y_train1, X_valid=X_valid1, y_valid=y_valid1, n_epochs=1000)
print(rnd_search.best_estimator_)       #best parameters
y_pred=rnd_search.predict(X_test1)
print(accuracy_score(y_test1, y_pred))
rnd_search.best_estimator_.save("./my_model_final.ckpt")

#8.4 add batch_normalization into the cv
from sklearn.model_selection import RandomizedSearchCV

param_distribs = {
    "n_neurons": [10, 30, 50, 70, 90, 100, 120, 140, 160],
    "batch_size": [10, 50, 100, 500],
    "learning_rate": [0.01, 0.02, 0.05, 0.1],
    "activation": [tf.nn.relu, tf.nn.elu],
    "batch_norm_momentum": [0.9, 0.95, 0.98, 0.99, 0.999],
}

rnd_search_bn = RandomizedSearchCV(DNNClassifier(random_state=42), param_distribs, n_iter=50, cv=3,
                                   random_state=42, verbose=2)
rnd_search_bn.fit(X_train1, y_train1, X_valid=X_valid1, y_valid=y_valid1, n_epochs=1000)

#8.4 add dropout into the cv
from sklearn.model_selection import RandomizedSearchCV

param_distribs = {
    "n_neurons": [10, 30, 50, 70, 90, 100, 120, 140, 160],
    "batch_size": [10, 50, 100, 500],
    "learning_rate": [0.01, 0.02, 0.05, 0.1],
    "activation": [tf.nn.relu, tf.nn.elu],
    "dropout_rate": [0.2, 0.3, 0.4, 0.5, 0.6],
}

rnd_search_dropout = RandomizedSearchCV(DNNClassifier(random_state=42), param_distribs, n_iter=50,cv=3, random_state=42, verbose=2)
rnd_search_dropout.fit(X_train1, y_train1, X_valid=X_valid1, y_valid=y_valid1, n_epochs=1000)

##exercise9
reset_graph()
restore_saver=tf.train.import_meta_graph("./my_model_final.ckpt.meta")
X=tf.get_default_graph().get_tensor_by_name("X:0")
y=tf.get_default_graph().get_tensor_by_name("y:0")
loss=tf.get_default_graph().get_tensor_by_name("loss:0")
y_proba=tf.get_default_graph().get_tensor_by_name("Y_proba:0")
logits=Y_proba.op.inputs[0]
accuracy=tf.get_default_graph().get_tensor_by_name("accuracy:0")

learning_rate=0.01
output_layer_vars=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="logits")
optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate,name="Adam2")
training_op=optimizer.minimize(loss,var_list=output_layer_vars)
correct=tf.nn.in_top_k(logits, y, 1)
accuracy=tf.reduce_mean(tf.cast(correct,tf.float32),name="accuracy")
init=tf.global_variables_initializer()
five_frozen_saver=tf.train.Saver()

#9.2
X_train2_full = X_train[y_train >= 5]
y_train2_full = y_train[y_train >= 5] - 5
X_valid2_full = X_valid[y_valid >= 5]
y_valid2_full = y_valid[y_valid >= 5] - 5
X_test2 = X_test[y_test >= 5]
y_test2 = y_test[y_test >= 5] - 5
def sample_n_instances_per_class(X, y, n=100):
    Xs, ys = [], []
    for label in np.unique(y):
        idx = (y == label)
        Xc = X[idx][:n]
        yc = y[idx][:n]
        Xs.append(Xc)
        ys.append(yc)
    return np.concatenate(Xs), np.concatenate(ys)
X_train2, y_train2 = sample_n_instances_per_class(X_train2_full, y_train2_full, n=100)
X_valid2, y_valid2 = sample_n_instances_per_class(X_valid2_full, y_valid2_full, n=30)

import time
n_epochs=100
batch_size=20
max_checks_without_progress=20
checks_without_progress=0
best_loss=np.infty
with tf.Session() as sess:
    init.run()
    restore_saver.restore(sess, "./my_model_final.ckpt")
    t0=time.time()
    for epoch in range(n_epochs):
        rnd_idx=np.random.permutation(len(X_train2))
        for rnd_indices in np.array_split(rnd_idx,len(X_train2)//batch_size):
            X_batch,y_batch=X_train2[rnd_indices],y_train2[rnd_indices]
            sess.run(training_op,feed_dict={X:X_batch,y:y_batch})
        loss_val,accuracy_val=sess.run([loss,accuracy],feed_dict={X:X_valid2,y:y_valid2})
        if loss_val<best_loss:
            checks_without_progress=0
            best_loss=loss_val
            save_path=five_frozen_saver.save(sess, "./my_mnist_model_5_to_9_five_frozen")
        else:
            checks_without_progress+=1
            if checks_without_progress>max_checks_without_progress:
                print("early stop")
                break
        print(loss_val,accuracy_val)
    t1=time.time()
    total_train_time=t1-t0
    print("total training time is ",total_train_time)
with tf.Session() as sess:
    five_frozen_saver.restore(sess, "./my_mnist_model_5_to_9_five_frozen")
    acc_test=accuracy.eval(feed_dict={X:X_valid2,y:y_valid2})
    print(acc_test)

#9.3

import time
n_epochs=100
batch_size=20
max_checks_without_progress=20
checks_without_progress=0
best_loss=np.infty

with tf.Session() as sess:
    init.run()
    restore_saver.restore(sess, "./my_model_final.ckpt")
    t0=time.time()
    h5_cache=sess.run(hidden5,feed_dict={X:X_train2})
    for epoch in range(n_epochs):
        rnd_idx=np.random.permutation(len(X_train2))
        for rnd_indices in array_split(rnd_idx,len(X_train2)//batch_size):
            h5_batch,y_batch=h5_cache[rnd_indices],y_train2[rnd_indices]
            sess.run(training_op,feed_dict={X:h5_batch,y:y_batch})
        loss_val,accuracy_val=sess.run([loss,accuracy],feed_dict={X:X_valid2,y:y_valid2})
        if loss_val<best_loss:
            checks_without_progress=0
            best_loss=loss_val
            save_path=five_frozen_saver.save(sess, "./my_mnist_model_5_to_9_five_frozen")
        else:
            checks_without_progress+=1
            if checks_without_progress>max_checks_without_progress:
                print("early stop")
                break
        print(loss_val,accuracy_val)
    t1=time.time()
    total_train_time=t1-t0
    print("total training time is ",total_train_time)
with tf.Session() as sess:
    five_frozen_saver.restore(sess, "./my_mnist_model_5_to_9_five_frozen")
    acc_test=accuracy.eval(feed_dict={X:X_valid2,y:y_valid2})
    print(acc_test)


#9.4
reset_graph()
n_outputs=5
restore_saver=tf.train.import_meta_graph("./my_model_final.ckpt.meta")
X=tf.get_default_graph().get_tensor_by_name("X:0")
y=tf.get_default_graph().get_tensor_by_name("y:0")
hidden4=tf.get_default_graph().get_tensor_by_name("hidden4:0")
logits=tf.layers.dense(hidden4, n_outputs,name="new_outputs")
with tf.name_scope("new_loss"):
    xentropy=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=logits,name="xentropy")
    loss=tf.reduce_mean(xentropy,name="loss")
with tf.name_scope("new_train"):
    train_vars=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="new_outputs")
    optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate,name="Adam2")
    training_op=optimizer.minimize(loss,var_list=train_vars)
with tf.name_scope("eval"):
    correct=tf.nn.in_top_k(logi, y, 1)
    accuracy=tf.reduce_mean(tf.cast(correct,tf.float32),name="accuracy")
init=tf.global_variables_initializer()
four_frozen_saver=tf.train.Saver()

n_epochs=100
batch_size=20
max_checks_without_progress=20
checks_without_progress=0
best_loss=np.infty

with tf.Session() as sess:
    init.run()
    restore_saver.restore(sess, "./my_model_final.ckpt")
    for epoch in range(n_epochs):
        rnd_idx=np.random.permutation(len(X_train2))
        for rnd_indices in np.array_split(rnd_idx,len(X_train2)//batch_size):
            X_batch,y_batch=X_train2[rnd_indices],y_train2[rnd_indices]
            sess.run(training_op,feed_dict={X:X_batch,y:y_batch})
        loss_val,accuracy_val=sess.run([loss,accuracy],feed_dict={X:X_batch,y:y_batch})
        if loss_val<best_loss:
            checks_without_progress=0
            best_loss=loss_val
            save_path=four_frozen_saver.save(sess, "./my_mnist_model_5_to_9_five_frozen")
        else:
            checks_without_progress+=1
            if checks_without_progress>max_checks_without_progress:
                print("early stop")
                break
        print(loss_val,accuracy_val)
    save_path=four_frozen_saver.save(sess, "./my_mnist_model_5_to_9_five_frozen")

with tf.Session() as sess:
    four_frozen_saver.restore(sess, "./my_mnist_model_5_to_9_five_frozen")
    acc_test=accuracy.eval(feed_dict={X:X_valid2,y:y_valid2})
    print(acc_test)

#9.5
learning_rate=0.01
unfrozen_vars=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="hidden[34]|new_outputs")
optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op=optimizer.minimize(loss,var_list=unfrozen_vars)

init=tf.global_variables_initializer()
unfrozen_top2=tf.train.Saver()

n_epochs=100
batch_size=20
max_checks_without_progress=20
checks_without_progress=0
best_loss=np.infty
with tf.Session() as sess:
    init.run()
    restore_saver.restore(sess, "./my_model_final.ckpt")
    for epoch in range(n_epochs):
        rnd_idx=np.random.permutation(len(X_train2))
        for rnd_indices in np.array_split(rnd_idx, len(X_train2)//batch_size):
            X_batch,y_batch=X_train2[rnd_indices],y_train2[rnd_indices]
            sess.run(training_op,feed_dict={X:X_batch,y:y_batch})
        loss_val,accuracy_val=sess.run([loss,accuracy],feed_dict={X:X_batch,y:y_batch})
        if loss_val<best_loss:
            checks_without_progress=0
            best_loss=loss_val
            save_path=unfrozen_top2.save(sess, "./my_mnist_model_5_to_9_five_frozen")
        else:
            checks_without_progress+=1
            if checks_without_progress>max_checks_without_progress:
                print("early stop")
                break
        print(loss_val,accuracy_val)
    save_path=unfrozen_top2.save(sess, "./my_mnist_model_5_to_9_five_frozen")

with tf.Session() as sess:
    unfrozen_top2.restore(sess, "./my_mnist_model_5_to_9_five_frozen")
    acc_test=accuracy.eval(feed_dict={X:X_valid2,y:y_valid2})
    print(acc_test)


##exercise10
#10.1
he_init=tf.variance_scaling_initializer()
def dnn(inputs,n_hidden_layers=5,n_neurons=100,name=None,activation=tf.nn.elu,initializer=he_init):
    with tf.name_scope(scope_name):
        for layer in range(n_hidden_layers):
            inputs=tf.layers.dense(inputs, n_neurons,activation=activation,kernel_initializer=initializer,name="hidden%d" % (layer+1))
        return inputs
dnn_a_outputs=dnn(X1,name="dnn_a")
dnn_b_outputs=dnn(X2,name="dnn_b")

reset_graph()
X=tf.placeholder(tf.float32,shape=(None,2,n_inputs),name="X")
X1,X2=tf.unstack(X,axis=1)
y=tf.placeholder(tf.int32,shape=[None,1],name="y")

dnn_outs=tf.concat([dnn_a_outputs,dnn_b_outputs], axis=1)
extra_hidden=tf.layers.dense(dnn_outs, units=10,activation=tf.nn.elu,kernel_initializer=he_init,name="extra_hidden")
logits=tf.layers.dense(extra_hidden, units=1,kernel_initializer=he_init,name="outputs")
y_proba=tf.nn.sigmoid(logits)
y_pred=tf.cast(tf.greater_equal(logits,0),tf.int32)
y_as_float=tf.cast(y, tf.float32)
xentropy=tf.nn.sigmoid_cross_entropy_with_logits(labels=y_as_float,logits=logits,name="xentropy")
loss=tf.reduce_mean(xentropy,name="loss")

learning_rate=0.01
momentum=0.95
optimizer=tf.train.MomentumOptimizer(learning_rate, momentum,use_nesterov=True)
training_op=optimizer.minimize(loss)

correct=tf.equal(y_pred, y)
accuracy=tf.reduce_mean(tf.cast(correct,tf.float32))

init=tf.global_variables_initializer()
saver=tf.train.Saver()

#10.2
X_train1 = X_train
y_train1 = y_train

X_train2 = X_valid
y_train2 = y_valid

X_test = X_test
y_test = y_test
def generate_batch(images, labels, batch_size):
    size1 = batch_size // 2
    size2 = batch_size - size1
    if size1 != size2 and np.random.rand() > 0.5:
        size1, size2 = size2, size1
    X = []
    y = []
    while len(X) < size1:
        rnd_idx1, rnd_idx2 = np.random.randint(0, len(images), 2)
        if rnd_idx1 != rnd_idx2 and labels[rnd_idx1] == labels[rnd_idx2]:
            X.append(np.array([images[rnd_idx1], images[rnd_idx2]]))
            y.append([1])
    while len(X) < batch_size:
        rnd_idx1, rnd_idx2 = np.random.randint(0, len(images), 2)
        if labels[rnd_idx1] != labels[rnd_idx2]:
            X.append(np.array([images[rnd_idx1], images[rnd_idx2]]))
            y.append([0])
    rnd_indices = np.random.permutation(batch_size)
    return np.array(X)[rnd_indices], np.array(y)[rnd_indices]

#10.3
X_test1,y_test1=generate_batch(X_test, y_test, batch_size=len(X_test))
n_epochs=100
batch_size=500
saver=tf.train.Saver()
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(len(X_train1)//batch_size):
            X_batch,y_batch=generate_batch(X_train1, y_train1, batch_size)
            loss_val,_=sess.run([loss,training_op],feed_dict={X:X_batch,y:y_batch})
        print(epoch,"loss is : ",loss_val)
        if epoch %5==0:
            acc_test=accuracy.eval(feed_dict={X:X_test1,y:y_test1})
            print(epoch,"testdata accuracy is :",acc_test)
    save_path=saver.save(sess, "./my_digit_comparison_model.ckpt")

#10.4
reset_graph()
restore_saver=tf.train.import_meta_graph("./my_digit_comparison_model.ckpt.meta")

n_inputs = 28 * 28  # MNIST
n_outputs = 10
X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int32, shape=(None), name="y")
dnn_a_outputs=tf.get_default_graph().get_tensor_by_name("dnn_a:0")
logits=tf.layers.dense(dnn_a_outputs, n_outputs,name="new_outputs")

with tf.name_scope("new_loss"):
    xentropy=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=logits,name="xentropy")
    loss=tf.reduce_mean(xentropy,name="loss")
with tf.name_scope("new_train"):
    train_vars=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="new_outputs")
    optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate,name="Adam2")
    training_op=optimizer.minimize(loss,var_list=train_vars)
with tf.name_scope("eval"):
    correct=tf.nn.in_top_k(logi, y, 1)
    accuracy=tf.reduce_mean(tf.cast(correct,tf.float32),name="accuracy")

init=tf.global_variables_initializer()
frozen_hidden_layers_saver=tf.train.Saver()

n_epochs = 100
batch_size = 50

with tf.Session() as sess:
    init.run()
    restore_saver.restore(sess, "./my_digit_comparison_model.ckpt")
    for epoch in range(n_epochs):
        rnd_idx=np.random.permutation(len(X_train2))
        for rnd_indices in np.array_split(rnd_idx,len(X_train2)//batch_size):
            X_batch,y_batch=X_train2[rnd_indices],y_train2[rnd_indices]
            sess.run(training_op,feed_dict={X:X_batch,y:y_batch})
        loss_val,accuracy_val=sess.run([loss,accuracy],feed_dict={X:X_test,y:y_test})
        print(loss_val,accuracy_val)
    save_path=frozen_hidden_layers_saver.save(sess, "./my_frozen_hidden_layers_model.ckpt")

