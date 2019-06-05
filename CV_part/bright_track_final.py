#!/usr/bin/python
# this script is used for video cut :one picture/per second
import os
import cv2
import shutil
import numpy as np
import pandas as pd
import sys
from PIL import Image
import numpy as np
import pylab as pl
import heapq



def video2frame(video_path, frame_width, frame_height, interval):
    """
    将视频按固定间隔读取写入图片
    :param video_name: 视频名字
    :param formats:　包含的所有视频格式
    :param frame_width:　保存帧宽
    :param frame_height:　保存帧高
    :param interval:　保存帧间隔
    :return:　帧图片
    """
    video = video_path
    print("正在读取视频：", video)
    video_name = video[:-4]
    video_save_full_path = os.path.join(video_name) + "/"
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
        #print ("---> 正在读取第%d帧:" % frame_index, success)

        if frame_index % interval == 0:
            resize_frame = cv2.resize(frame, (frame_width, frame_height), interpolation=cv2.INTER_AREA)
                # cv2.imwrite(each_video_save_full_path + each_video_name + "_%d.jpg" % frame_index, resize_frame)
            cv2.imwrite(video_save_full_path + "%d.jpg" %frame_count, resize_frame)
            frame_count += 1
        frame_index += 1
    cap.release()


def get_loc_list(pictures):
    loc_list=[]
    for picture in pictures:
        print(picture,end='\t')
        (x,y)=get_picture_loc(picture)
        loc_list.append((x,y))
    return loc_list

def get_picture_loc(name):
    im=np.array(Image.open(name).convert('L'))          #PIL Image 把图片从RGB转换灰度图，再转换为 numpy 数组
    pl.gray()
    B1=np.float64(im)                                   #像素值转换为float值
    #print(B1.shape)                                     #show the picture dpi  (1440*1080) 
    B2=np.where(B1>250,1,0)                             #设定阈值，转换为0-1矩阵
    C=list(np.sum(B2,axis=0)) 
    D=list(np.sum(B2,axis=1))
    Xmax1=heapq.nlargest(1,C)                           #找出最大的x，y值为坐标
    x=C.index(Xmax1)
    Ymax1=heapq.nlargest(1,D)
    y=D.index(Ymax1) 
    print("光点的坐标是 : (",x,",",y,")")
    return (x,y)

def caculate_dist(x1,x2,y1,y2,):
    if (x1==0 and y1==0) or (x2==0 or y2==0):
        move_distance=50
    else:
        move_distance=((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
    if move_distance <= 10 :
        move_distance=0
    return move_distance


def split_photo_per_scene(video_dir,sensor_time_file):
    scene_df=pd.read_csv(sensor_time_file,sep='\t',header=0)
    scene_df['per_scene_photos']=pd.Series(scene_df['spend_ms']/1000,dtype=int)
    scene_df['end_photo']=scene_df['per_scene_photos'].cumsum()
    scene_df['start_photo']=scene_df['end_photo'] - scene_df['per_scene_photos']
    scene_df=scene_df[['sample_id','scene','start_photo','end_photo']]
    os.chdir(video_dir)
    scene_dist=[]
    move_count_array=[]
    for row_index,row in scene_df.iterrows():
        scene_index=row_index+1
        scene_name="scene"+str(scene_index)
        if os.path.exists(scene_name):
            shutil.rmtree(scene_name)
            os.makedirs(scene_name)
        else:
            os.makedirs(scene_name)
        for i in range(row[2],row[3]):
            file_name=str(i)+".jpg"
            shutil.move(file_name, scene_name)
        loc_list=[]
        pictures = os.listdir(scene_name)
        pictures.sort(key=lambda x:int(x[:-4]))
        os.chdir(scene_name)
        location_list=get_loc_list(pictures)

        count=0
        move_count=0
        move_distances_array=[]
        for count in range(len(location_list)-1):
            move_dist=caculate_dist(location_list[count][0],location_list[count+1][0] , location_list[count][1], location_list[count+1][1])
            if move_dist>30:
                move_count=move_count+1
            move_distances_array.append(move_dist)
        move_count_array.append(move_count)
        scene_dist.append(round(sum(move_distances_array),3))
        print(scene_name,"move_count",move_count)
        print(scene_name,"total_movement : ",sum(move_distances_array))
        os.chdir("../")
    os.chdir("../")
    return move_count_array,scene_dist,scene_df


video_path=sys.argv[1]
sensor_time_file=sys.argv[2]
video_dir=video_path[:-4]
if os.path.exists(video_dir):
    shutil.rmtree(video_dir)
    os.makedirs(video_dir)
else:
    os.makedirs(video_dir)

video2frame(video_path, 320, 240, 30)
move_count_list,scene_dist_list,scene_df=split_photo_per_scene(video_dir, sensor_time_file)
print(scene_dist_list)
print(scene_df)
scene_df=scene_df[['sample_id','scene']]
scene_df['cv_movement']=pd.Series(scene_dist_list)
scene_df['cv_move_count']=pd.Series(move_count_list)
print(scene_df)
per_scene_cv_movement_txt=sensor_time_file[:-4]+"_per_scene.txt"

scene_df.to_csv(per_scene_cv_movement_txt,columns=['sample_id','scene','cv_movement','cv_move_count'],sep="\t",index=False)
total_result=sensor_time_file[:-4]+"\ttotal_cv"+"\t"+str(round(sum(scene_dist_list),3))+"\t"+str(sum(move_count_list))
my_open = open(per_scene_cv_movement_txt, 'a')
my_open.write(total_result)
my_open.close()
