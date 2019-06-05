import pandas as pd 
import numpy as np
import tensorflow as tf
from tensorflow import keras
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

##数据预处理
boston_housing = keras.datasets.boston_housing

(train_data, train_labels), (test_data, test_labels) = boston_housing.load_data()

# Shuffle the training set
order = np.argsort(np.random.random(train_labels.shape))
train_data = train_data[order]
train_labels = train_labels[order]
import pandas as pd

column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
                'TAX', 'PTRATIO', 'B', 'LSTAT']

df = pd.DataFrame(train_data, columns=column_names)
df.head()
mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
train_data = (train_data - mean) / std
test_data = (test_data - mean) / std

#模型参数
n_inputs=train_data.shape[1]
reset_graph()
X=tf.placeholder(tf.float32,shape=(None,n_inputs),name="X")
y=tf.placeholder(tf.int32,shape=(None),name="y")
n_hidden1=100
n_hidden2=50
n_outputs=1
learning_rate=0.01
n_epochs=1000

#构建图谱
with tf.name_scope("dnn"):
    hidden1=tf.layers.dense(X,n_hidden1,activation=tf.nn.relu,name="hidden1")
    hidden2=tf.layers.dense(hidden1,n_hidden2,activation=tf.nn.relu,name="hidden2")
    output=tf.layers.dense(hidden2,n_outputs,activation=None,name="outputs")

with tf.name_scope("loss"):
    loss=tf.losses.mean_squared_error(labels=y,predictions=output)

with tf.name_scope("train"):
    optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    training_op=optimizer.minimize(loss)

init=tf.global_variables_initializer()
saver=tf.train.Saver()

#执行会话
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        sess.run(training_op,feed_dict={X:train_data,y:train_labels})
        if (epoch+1) % 100 ==0:
            loss_val=loss.eval(feed_dict={X:train_data,y:train_labels})
            print("epoch is ",epoch+1,"and loss_value is ",loss_val)
            save_path=saver.save(sess,"./my_model.ckpt",global_step=(epoch+1))
    save_path=saver.save(sess,"./my_final_model.ckpt")
#对测试卷预测
with tf.Session() as sess:
    saver.restore(sess,"./my_final_model.ckpt")
    y_pred=output.eval(feed_dict={X:test_data})
print(y_pred[:10])
print(test_labels[:10])
