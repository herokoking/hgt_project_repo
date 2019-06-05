#!/usr/bin/python3
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, make_scorer
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
%matplotlib inline
from datetime import datetime
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict, cross_val_score, cross_validate


def generate_samples(df,n):
	df_mean_std=df.groupby('group').agg(['mean','std'])
	df_mean_std=round(df_mean_std,3)
	z=df_mean_std.columns.get_level_values(1)
	for x in df_mean_std.index:
		dict_name=str(x)+'_dict'
		dict_name={}
		for y in df_mean_std.columns.get_level_values(0):
			dict_name[y]=pd.Series(np.random.normal(df_mean_std.loc[x,y][z[0]],df_mean_std.loc[x,y][z[1]],n))
		dict_name['group']=x
		df=pd.concat([df,round(pd.DataFrame(dict_name),3)],axis=0)
	return df

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

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

def transformat_define(df):
	df=np.array(df)
	return df
def fpr_score(y_true, y_pred):
    model_confuse_matrix = confusion_matrix(y_true, y_pred)
    tp = model_confuse_matrix[0, 0]
    tn = model_confuse_matrix[1, 1]
    fn = model_confuse_matrix[0, 1]
    fp = model_confuse_matrix[1, 0]
    if tn + fp == 0:
        fpr_value = 0
    else:
        specificity = tn / (tn + fp)
        fpr_value = 1 - specificity
    return fpr_value
def plot_confusion_matrix(confusion_mat):
    plt.imshow(confusion_mat,interpolation='nearest',cmap=plt.cm.Paired)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks=np.arange(2)
    plt.xticks(tick_marks,tick_marks)
    plt.yticks(tick_marks,tick_marks)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def performance_model(x):
    x=x.astype(np.str)
    x[0,0]="tp : " + x[0,0]
    x[1,1]="tn : " + x[1,1]
    x[0,1]="ft : " + x[0,1]
    x[1,0]="fp : " + x[1,0]
    print(pd.DataFrame(x,index=['True_ADHD','True_Normal'],columns=['Predict_ADHD','Predict_Normal']))



eeg_df = pd.read_csv('input_materials_20190305.txt', sep='\t')
eeg_df.head()
eeg_df=eeg_df.iloc[:,1:]
filted_df=generate_samples(eeg_df,489).reset_index()
y=filted_df['group']
X=filted_df.drop(columns=['index','group'])
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
X_valid, X_train = X_train[:100], X_train[100:]
y_valid, y_train = y_train[:100], y_train[100:]
X_train,X_valid,X_test=transformat_define(X_train),transformat_define(X_valid),transformat_define(X_test)
y_train,y_valid,y_test=transformat_define(y_train),transformat_define(y_valid),transformat_define(y_test)

#define parameters
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)

n_inputs=X_train.shape[1]
n_hidden1=30
n_hidden2=20
n_hidden3=10
n_outputs=2
batch_size=50
learning_rate=0.01
n_epochs=1000
max_checks_without_progress = 50
checks_without_progress = 0
best_loss = np.infty


#build compute graph
reset_graph()
X=tf.placeholder(tf.float32,shape=(None,n_inputs),name="X")
y=tf.placeholder(tf.int32,shape=(None),name="y")


with tf.name_scope("dnn"):
    hidden1=tf.layers.dense(X, n_hidden1,activation=tf.nn.relu,name="hidden1")
    hidden2=tf.layers.dense(hidden1, n_hidden2,activation=tf.nn.relu,name="hidden2")
    hidden3=tf.layers.dense(hidden2,n_hidden3,activation=tf.nn.relu,name="hidden3")
    logits=tf.layers.dense(hidden3, n_outputs,name="outputs")
with tf.name_scope("loss"):
    xentropy=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=logits)
    loss=tf.reduce_mean(xentropy,name="loss")
with tf.name_scope("train"):
    optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op=optimizer.minimize(loss)
with tf.name_scope("eval"):
    correct=tf.nn.in_top_k(logits,y, 1)
    accuracy=tf.reduce_mean(tf.cast(correct, tf.float32))
y_pred=tf.nn.softmax(logits)
init=tf.global_variables_initializer()
saver=tf.train.Saver()
loss_summary = tf.summary.scalar('entropy', loss)
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for X_batch,y_batch in shuffle_batch(X_train, y_train, batch_size):
            training_op.run(feed_dict={X:X_batch,y:y_batch})
        loss_value=loss.eval(feed_dict={X:X_valid,y:y_valid})
        if loss_value < best_loss:
            best_loss=loss_value
            save_path=saver.save(sess, "./my_model_final.ckpt")
            checks_without_progress=0
        else:
            checks_without_progress+=1
            if checks_without_progress>max_checks_without_progress:
                print("early stopping")
                break
        if (epoch+1)%10==0:
        	accuracy_value=accuracy.eval(feed_dict={X:X_valid,y:y_valid})
        	print("epoch is :",epoch+1,"accuracy :",accuracy_value,"loss_value is : ",loss_value)
        summary_str = loss_summary.eval(feed_dict={X: X_batch, y: y_batch})
        file_writer.add_summary(summary_str, epoch)
with tf.Session() as sess:
    saver.restore(sess, "./my_model_final.ckpt")
    accuracy_testdata=accuracy.eval(feed_dict={X:X_test,y:y_test})
    #print("Final test accuracy: {:.2f}%".format(accuracy_testdata*100))
    y_prob=y_pred.eval(feed_dict={X:X_test})
    y_predict=y_prob.argmax(axis=1)
    testdataset_fpr = fpr_score(y_test, y_predict) * 100
    testdataset_accuracy = accuracy_score(y_test, y_predict) * 100
file_writer.close()
performance_model(confusion_matrix(y_test,y_predict))
print("测试数据集误诊率为 : %0.2f%%" % testdataset_fpr, sep='\t')
print("测试数据集准确率为 : %0.2f%%" % testdataset_accuracy, sep='\t')



##use keras to build the graph
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, make_scorer
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
%matplotlib inline
from datetime import datetime
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict, cross_val_score, cross_validate
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
def generate_samples(df,n):
    df_mean_std=df.groupby('group').agg(['mean','std'])
    df_mean_std=round(df_mean_std,3)
    z=df_mean_std.columns.get_level_values(1)
    for x in df_mean_std.index:
        dict_name=str(x)+'_dict'
        dict_name={}
        for y in df_mean_std.columns.get_level_values(0):
            dict_name[y]=pd.Series(np.random.normal(df_mean_std.loc[x,y][z[0]],df_mean_std.loc[x,y][z[1]],n))
        dict_name['group']=x
        df=pd.concat([df,round(pd.DataFrame(dict_name),3)],axis=0)
    return df
def transformat_define(df):
    df=np.array(df)
    return df


model=Sequential()

eeg_df = pd.read_csv('input_materials_20190305.txt', sep='\t')
eeg_df.head()
eeg_df=eeg_df.iloc[:,1:]
filted_df=generate_samples(eeg_df,489).reset_index()
y=filted_df['group']
X=filted_df.drop(columns=['index','group'])
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
X_valid, X_train = X_train[:100], X_train[100:]
y_valid, y_train = y_train[:100], y_train[100:]
X_train,X_valid,X_test=transformat_define(X_train),transformat_define(X_valid),transformat_define(X_test)
y_train,y_valid,y_test=transformat_define(y_train),transformat_define(y_valid),transformat_define(y_test)
X_train=X_train.astype('float32')
X_test=X_test.astype('float32')
onehot_train = keras.utils.to_categorical(y_train, num_classes=2)
onehot_test=keras.utils.to_categorical(y_test,num_classes=2)

model.add(Dense(30,activation='relu',input_shape=(72,)))
model.add(Dense(20,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(2,activation='softmax'))
model.summary()
from keras.utils.vis_utils import plot_model
plot_model(model,to_file="keras_dnn.png")
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
SVG(model_to_dot(model).create(prog="dot",format='svg'))

from keras import optimizers
sgd=optimizers.SGD(lr=0.01,momentum=0.9)
model.compile(optimizer=sgd,loss="categorical_crossentropy",metrics=['accuracy'])
model.fit(x=X_train,y=onehot_train,epochs=50,batch_size=20)
model.evaluate(X_test,onehot_test,batch_size=128)