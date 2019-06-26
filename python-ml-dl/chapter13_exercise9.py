#!/usr/bin/python3
# this script is used for chapter13_exercise9 ---- Transfer learning for large image classification
import sys
import tarfile
from six.moves import urllib
import os

TF_MODELS_URL = "http://download.tensorflow.org/models"
INCEPTION_V3_URL = TF_MODELS_URL + "/inception_v3_2016_08_28.tar.gz"
INCEPTION_PATH = os.path.join("datasets", "inception")
INCEPTION_V3_CHECKPOINT_PATH = os.path.join(INCEPTION_PATH, "inception_v3.ckpt")

def download_progress(count, block_size, total_size):
    percent = count * block_size * 100 // total_size
    sys.stdout.write("\rDownloading: {}%".format(percent))
    sys.stdout.flush()

def fetch_pretrained_inception_v3(url=INCEPTION_V3_URL, path=INCEPTION_PATH):
    if os.path.exists(INCEPTION_V3_CHECKPOINT_PATH):
        return
    os.makedirs(path, exist_ok=True)
    tgz_path = os.path.join(path, "inception_v3.tgz")
    urllib.request.urlretrieve(url, tgz_path, reporthook=download_progress)
    inception_tgz = tarfile.open(tgz_path)
    inception_tgz.extractall(path=path)
    inception_tgz.close()
    os.remove(tgz_path)

fetch_pretrained_inception_v3()

FLOWERS_URL = "http://download.tensorflow.org/example_images/flower_photos.tgz"
FLOWERS_PATH = os.path.join("datasets", "flowers")


def fetch_flowers(url=FLOWERS_URL, path=FLOWERS_PATH):
    if os.path.exists(FLOWERS_PATH):
        return
    os.makedirs(path, exist_ok=True)
    tgz_path = os.path.join(path, "flower_photos.tgz")
    urllib.request.urlretrieve(url, tgz_path, reporthook=download_progress)
    flowers_tgz = tarfile.open(tgz_path)
    flowers_tgz.extractall(path=path)
    flowers_tgz.close()
    os.remove(tgz_path)
def download_progress(count, block_size, total_size):
    percent = count * block_size * 100 // total_size
    sys.stdout.write("\rDownloading: {}%".format(percent))
    sys.stdout.flush()
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

fetch_flowers()

flowers_root_path = os.path.join(FLOWERS_PATH, "flower_photos")
flower_classes = sorted([dirname for dirname in os.listdir(flowers_root_path)
                  if os.path.isdir(os.path.join(flowers_root_path, dirname))])
flower_classes

from collections import defaultdict

image_paths = defaultdict(list)

image_paths

for flower_class in flower_classes:
    image_dir = os.path.join(flowers_root_path, flower_class)
    for filepath in os.listdir(image_dir):
        if filepath.endswith(".jpg"):
            image_paths[flower_class].append(os.path.join(image_dir, filepath))

for paths in image_paths.values():
    paths.sort()   

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
n_examples_per_class=2
channels=3

for flower_class in flower_classes:
    print("Class:", flower_class)
    plt.figure(figsize=(10,5))
    for index, example_image_path in enumerate(image_paths[flower_class][:n_examples_per_class]):
        example_image = mpimg.imread(example_image_path)[:, :, :channels]
        plt.subplot(100 + n_examples_per_class * 10 + index + 1)
        plt.title("{}x{}".format(example_image.shape[1], example_image.shape[0]))
        plt.imshow(example_image)
        plt.axis("off")
    plt.show()

import tensorflow as tf
def prepare_image_with_tensorflow(image,target_width=299,target_height=299,max_zoom=0.2):
	image_shape=tf.cast(tf.shape(image), tf.float32)
	height=image_shape[0]
	width=image_shape[1]
	image_ratio=width/height
	target_image_ratio=target_width/target_height
	crop_vertically=image_ratio<target_image_ratio
	crop_width=tf.cond(crop_vertically,lambda:width,lambda:height*target_image_ratio)
	crop_height=tf.cond(crop_vertically,lambda:width*target_image_ratio,lambda:height)
	resize_factor=tf.random_uniform(shape=[], minval=1.0,maxval=1.0+max_zoom)
	crop_width=tf.cast(crop_width/resize_factor,tf.int32)
	crop_height=tf.cast(crop_height/resize_factor,tf.int32)
	box_size=tf.stack([crop_height,crop_width,3])
	image=tf.random_crop(image, box_size)
	image=tf.image.random_flip_left_right(image)
	image_batch=tf.expand_dims(image, 0)
	image_batch=tf.image.resize_bilinear(image_batch, [target_height,target_width])
	image=image_batch[0]/255
	return image

reset_graph()
input_image=tf.placeholder(tf.uint8,shape=[None,None,3])
prepare_image_op=prepare_image_with_tensorflow(input_image)
'''
#test a example photo
with tf.Session() as sess:
	prepare_image=prepare_image_op.eval(feed_dict={input_image:example_image})
plt.figure(figsize=(6,6))
plt.imshow(prepare_image)
plt.title("{}x{}".format(prepare_image.shape[1],prepare_image.shape[0]))
plt.axis("off")
plt.show()
'''

##use pretrained inception v3 model
from tensorflow.contrib.slim.nets import inception
import tensorflow.contrib.slim as slim
reset_graph()
X=tf.placeholder(tf.float32,shape=[None,height,width,channels],name="X")
training=tf.placeholder_with_default(False, shape=[])
with slim.arg_scope(inception.inception_v3_arg_scope()):
	logits,end_points=inception.inception_v3(X,num_classes=1001,is_training=training)
inception_saver=tf.train.Saver()
prelogits = tf.squeeze(end_points["PreLogits"], axis=[1, 2])			#或者用prelogits=tf.layers.flatten(end_points["PreLogits"])代替
n_outputs=len(flower_classes)
with tf.name_scope("new_output_layer"):
	flower_logits=tf.layers.dense(prelogits,n_outputs)
	Y_proba=tf.nn.softmax(flower_logits,name='Y_proba')
y=tf.placeholder(tf.int32,shape=[None])
with tf.name_scope("train"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=flower_logits, labels=y)
    loss = tf.reduce_mean(xentropy)
    optimizer = tf.train.AdamOptimizer()
    flower_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="flower_logits")
    training_op = optimizer.minimize(loss, var_list=flower_vars)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(flower_logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

with tf.name_scope("init_and_save"):
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()


#2019/6/25   做到9.4
flower_class_ids = {flower_class: index for index, flower_class in enumerate(flower_classes)}
flower_class_ids
flower_paths_and_classes = []
for flower_class, paths in image_paths.items():
    for path in paths:
        flower_paths_and_classes.append((path, flower_class_ids[flower_class]))

test_ratio = 0.2
train_size = int(len(flower_paths_and_classes) * (1 - test_ratio))

np.random.shuffle(flower_paths_and_classes)

flower_paths_and_classes_train = flower_paths_and_classes[:train_size]
flower_paths_and_classes_test = flower_paths_and_classes[train_size:]

from random import sample

def prepare_batch(flower_paths_and_classes, batch_size):
    batch_paths_and_classes = sample(flower_paths_and_classes, batch_size)
    images = [mpimg.imread(path)[:, :, :channels] for path, labels in batch_paths_and_classes]
    prepared_images = [prepare_image(image) for image in images]
    X_batch = 2 * np.stack(prepared_images) - 1 # Inception expects colors ranging from -1 to 1
    y_batch = np.array([labels for path, labels in batch_paths_and_classes], dtype=np.int32)
    return X_batch, y_batch
X_test, y_test = prepare_batch(flower_paths_and_classes_test, batch_size=len(flower_paths_and_classes_test))

n_epochs = 10
batch_size = 40
n_iterations_per_epoch = len(flower_paths_and_classes_train) // batch_size
with tf.Session() as sess:
    init.run()
    inception_saver.restore(sess,INCEPTION_V3_CHECKPOINT_PATH)
    for epoch in range(n_epochs):
        print("epoch",epoch,end="")
        for iteration in range(n_iterations_per_epoch):
            print(".",end="")
            X_batch,y_batch=prepare_batch(flower_paths_and_classes, batch_size)
            sess.run(training_op,feed_dict={X:X_batch,y:y_batch})
        acc_batch=accuracy.eval(feed_dict={X:X_batch,y_batch})
        print("last batch accuracy is : ", acc_batch)
        save_path=saver.save(sess, "./my_flowers_model")

#把测试集切割成10份，对每一份求准确度，然后再求十份的均值代表整个测试集的准确度
n_test_batches = 10
X_test_batches = np.array_split(X_test, n_test_batches)
y_test_batches = np.array_split(y_test, n_test_batches)

with tf.Session() as sess:
    saver.restore(sess, "./my_flowers_model")

    print("Computing final accuracy on the test set (this will take a while)...")
    acc_test = np.mean([
        accuracy.eval(feed_dict={X: X_test_batch, y: y_test_batch})
        for X_test_batch, y_test_batch in zip(X_test_batches, y_test_batches)])
    print("Test accuracy:", acc_test)
