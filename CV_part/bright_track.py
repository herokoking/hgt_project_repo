#!/usr/bin/python
# this script is used for video cut :one picture/per second
import os
import cv2
import shutil
videos_path = "./"
video_formats = [".MP4", ".MOV"]
frames_save_path = "./"
width = 320
height = 240
time_interval = 29
import numpy as np
import pandas as pd

def video2frame(video_path, formats, frame_save_path, frame_width, frame_height, interval):
    """
    将视频按固定间隔读取写入图片
    :param video_path: 视频存放路径
    :param formats:　包含的所有视频格式
    :param frame_save_path:　保存路径
    :param frame_width:　保存帧宽
    :param frame_height:　保存帧高
    :param interval:　保存帧间隔
    :return:　帧图片
    """
    video = video_path
    print("正在读取视频：", video)
    video_name = video[:-4]
    shutil.rmtree(video_name)
    os.mkdir(frame_save_path + video_name)
    video_save_full_path = os.path.join(frame_save_path, video_name) + "/"
    cap = cv2.VideoCapture(video)
    frame_index = 0
    frame_count = 0
    if cap.isOpened():
        success = True
    else:
        success = False
        print("读取失败!")

    while(success):
        success, frame = cap.read()
        print ("---> 正在读取第%d帧:" % frame_index, success)

        if frame_index % interval == 0:
            resize_frame = cv2.resize(frame, (frame_width, frame_height), interpolation=cv2.INTER_AREA)
                # cv2.imwrite(each_video_save_full_path + each_video_name + "_%d.jpg" % frame_index, resize_frame)
            cv2.imwrite(video_save_full_path + "%d.jpg" %frame_count, resize_frame)
            frame_count += 1
        frame_index += 1
    cap.release()

video2frame("./brightlight_move.mp4", [".mp4"], "./", 320, 240, 29)


from PIL import Image
import numpy as np
import pylab as pl
import heapq

def get_picture_loc(name):
    name="./brightlight_move/"+name
    im=np.array(Image.open(name).convert('L'))          #PIL Image 把图片从RGB转换灰度图，再转换为 numpy 数组
    pl.gray()
    B1=np.float64(im)                                   #像素值转换为float值
    #print(B1.shape)                                     #show the picture dpi  (1440*1080) 
    B2=np.where(B1>248,1,0)                             #设定阈值，转换为0-1矩阵
    C=list(np.sum(B2,axis=0)) 
    D=list(np.sum(B2,axis=1))
    Xmax1=heapq.nlargest(1,C)                           #找出最大的x，y值为坐标
    x=C.index(Xmax1)
    Ymax1=heapq.nlargest(1,D)
    y=D.index(Ymax1) 
    print("光点的坐标是 : (",x,",",y,")")
    loc_list.append((x,y))

def caculate_dist(x1,x2,y1,y2,):
    move_distance=((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
    return move_distance

def get_loc_list(pictures):
    for picture in pictures:
        print(picture,end='\t')
        get_picture_loc(picture)
    return loc_list

loc_list=[]
pictures = os.listdir("./brightlight_move/")
location_list=get_loc_list(pictures)

count=0
move_count=0
move_distances_array=[]
for count in range(len(location_list)-1):
    move_dist=caculate_dist(location_list[count][0],location_list[count+1][0] , location_list[count][1], location_list[count+1][1])
    if move_dist>50:
        move_count=move_count+1
    move_distances_array.append(move_dist)

print(move_count,"total_movement : ",sum(move_distances_array))
