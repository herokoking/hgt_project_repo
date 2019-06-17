#!/usr/bin/python3
#this script is used to record data enhance method in image processing
#usual data enhance method in image field include: 图片比例缩放，图片随机位置截取，图片随机水平和竖直翻转，图片随机角度旋转，图片亮度、对比度、颜色随机变化

import urllib
from PIL import Image
import tensorflow as tf
import numpy as np
import pandas as pd

#download the picture
im_url='https://image.ibb.co/fFxt7c/cat.png'
im=Image.open(urllib.request.urlopen(im_url))
im=tf.constant(np.array(im,dtype=np.uint8))
##图片比例缩放
resize=tf.image.resize_images(im,[100,200],method=tf.image.ResizeMethod.BILINEAR)
with tf.Session() as sess:
    im2=sess.run(resize)
Image.fromarray(np.uint8(im2))		#重现缩放后的图片
print(im.shape,im2.shape)

##图片随机截取
random_cropped_im1=tf.random_crop(im, [100,100,3])
random_cropped_im2=tf.random_crop(im, [150,100,3])
central_cropped_im=tf.image.central_crop(im, 1/3)
with tf.Session() as sess:
	random_im1,random_im2,central_im=sess.run([random_cropped_im1,random_cropped_im2,central_cropped_im])
Image.fromarray(np.uint8(random_im1))
Image.fromarray(np.uint8(random_im2))
Image.fromarray(np.uint8(central_im))

##图片随机水平和竖直翻转
h_flip=tf.image.random_flip_left_right(im)
v_flip=tf.image.random_flip_up_down(im)
with tf.Session() as sess:
	h_flip_im,v_flip_im=sess.run([h_flip,v_flip])
Image.fromarray(np.uint8(h_flip_im))
Image.fromarray(np.uint8(v_flip_im))

##图片随机角度旋转
with tf.variable_scope('random_rotate'):
	angle=tf.random_uniform([],minval=-45,maxval=45,dtype=tf.float32,name='random_angle')
	random_rotated_im=tf.contrib.image.rotate(im,angle)
with tf.Session() as sess:
	rotated_im=sess.run(random_rotated_im)
Image.fromarray(np.uint8(rotated_im))

##图片亮度、对比度、颜色改变
change_brihtness_im=tf.image.random_brightness(im, max_delta=1)
change_contrast_im=tf.image.random_contrast(im, lower=0, upper=3)
change_hue_im=tf.image.random_hue(im, max_delta=0.5)
with tf.Session() as sess:
	brightness_im,contrast_im,hue_im=sess.run([change_brihtness_im,change_contrast_im,change_hue_im])
Image.fromarray(np.uint8(brightness_im))
Image.fromarray(np.uint8(contrast_im))
Image.fromarray(np.uint8(hue_im))

def im_aug(im):
	im=tf.image.resize_images(im, [100,200])
	im=tf.image.random_flip_left_right(im)
	im=tf.random_crop(im, [96,96,3])
	im=tf.image.random_brightness(im, max_delta=0.5)
	im=tf.image.random_contrast(im, lower=0, upper=0.5)
	im=tf.image.random_hue(im, max_delta=0.5)
	return im
aug_im=im_aug(im)
import matplotlib.pyplot as plt
%matplotlib inline
nrows,ncols=3,3
figsize=(8,8)
_,figs=plt.subplots(nrows,ncols,figsize=figsize)
with tf.Session() as sess:
	for i in range(nrows):
		for j in range(ncols):
			aug_result=np.uint8(sess.run(aug_im))
			figs[i][j].imshow(aug_result)
			figs[i][j].axes.get_xaxis().set_visible(False)
			figs[i][j].axes.get_yaxis().set_visible(False)
	plt.show()
