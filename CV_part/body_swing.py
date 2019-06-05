#!/usr/bin/python
from PIL import Image
import numpy as np
import pylab as pl
import heapq
name="bright_point.jpg"
im=np.array(Image.open(name).convert('L'))			#PIL Image 把图片从RGB转换灰度图，再转换为 numpy 数组
pl.gray()
B1=np.float64(im)									#像素值转换为float值
print(B1.shape)										#show the picture dpi  (1440*1080) 
B2=np.where(B1>248,1,0)								#设定阈值，转换为0-1矩阵
C=list(np.sum(B2,axis=0)) 
D=list(np.sum(B2,axis=1))
Xmax1=heapq.nlargest(1,C) 							#找出最大的x，y值为坐标
x=C.index(Xmax1)
Ymax1=heapq.nlargest(1,D)
y=D.index(Ymax1) 
print("这一点的x、y坐标分别为",x,y) 
print(B1[y,x])
pl.imshow(B2)										#回查该点的像素值，若有多点情况用
pl.show()											#重现处理完的图片
