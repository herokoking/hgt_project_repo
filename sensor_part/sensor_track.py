#!/usr/bin/python3
#this script is used for spllit the sample_array into different part
#Usage: python sensor_track.py sensorTime_test04.txt test04.txt
import numpy as np
import pandas as pd
import sys
sensor_time_file=sys.argv[1]
sensor_raw_data_file=sys.argv[2]
per_scene_movement_txt=sensor_raw_data_file[:-4]+"_per_scene_movement.txt"
per_scene_movement_png=sensor_raw_data_file[:-4]+"_per_scene_movement.png"
scene_df=pd.read_csv(sensor_time_file,sep='\t',header=0)
scene_df['per_scene_counts']=pd.Series(scene_df['spend_ms']/20,dtype=int)
scene_df['end_sample']=scene_df['per_scene_counts'].cumsum()
scene_df['start_sample']=scene_df['end_sample'] - scene_df['per_scene_counts']
scene_df=scene_df[['sample_id','scene','start_sample','end_sample']]

def split_raw_acc_array_data(acc_df):
	sub_array_list=[]
	for row_index,row in scene_df.iterrows():
		sub_array_list.append(acc_df.iloc[row[2]:row[3],:])


	return sub_array_list

#load data in
from scipy import integrate
import matplotlib.pyplot as plt

acc_df=pd.read_csv(sensor_raw_data_file,sep='\t',header=1)
acc_df['x_acc']=abs(acc_df['ax(g)']-2*(acc_df['q1']*acc_df['q3']-acc_df['q0']*acc_df['q2']))
acc_df['y_acc']=abs(acc_df['ay(g)']-2*(acc_df['q0']*acc_df['q1']+acc_df['q2']*acc_df['q3']))
acc_df['z_acc']=abs(acc_df['az(g)']-(acc_df['q0']*acc_df['q0']-acc_df['q1']*acc_df['q1']-acc_df['q2']*acc_df['q2']+acc_df['q3']*acc_df['q3']))
acc_df=acc_df.iloc[:,-3:]
acc_df=acc_df.fillna(method="ffill")

#acc_df = acc_df.rename(columns={'Time(s)': 'time'})
#acc_df['time']=acc_df['time'].str.slice(0,-4)
def normalize_G(data_list):
    data_list_2s = data_list[:100]
    data_mean = data_list_2s.mean()
    return data_mean

#acc_df=acc_df-normalize_G(acc_df)
sub_acc_array_list=split_raw_acc_array_data(acc_df)


#algorithm  (acc2S_value)

def normalize_array_value(input_array):
    array_normalized_value = []
    for i in range(len(input_array)):
        if i == 0:
            array_normalized_value.append(input_array[i])
        else:
            array_normalized_value.append(input_array[i] - input_array[i - 1])
    return(array_normalized_value)

def acc2S_value(loc_a_value):
	v_int = integrate.cumtrapz(loc_a_value, x=None, dx=0.02, axis=-1, initial=None)
	v_value_array = normalize_array_value(v_int)
	S_int = integrate.cumtrapz(v_value_array, x=None,dx=0.02, axis=-1, initial=None)
	S_value_array = normalize_array_value(S_int)
	#plt.figure(12, facecolor = 'gray')
	#plt.subplot(221)
	#plt.plot(loc_a_value, 'r-')
	#plt.subplot(222)
	#plt.plot(v_value_array,'b-')
	#plt.subplot(212)
	#plt.plot(S_value_array,'g-')
# s_value的值就代表这一秒钟的位移
	s_value = integrate.trapz(v_value_array, x=None, dx=1 / 48, axis=-1)
	return s_value

def fillna_with_0(S_array):
	S_array=np.array(S_array)
	where_are_nan = np.isnan(S_array)
	S_array[where_are_nan] = 0
	return S_array

def distance_combine(x,y,z):
	distance=np.sqrt(x*x+y*y+z*z)
	return distance

def construct_S(X_s_value_array,Y_s_value_array,Z_s_value_array):
	X_s_value_array=fillna_with_0(X_s_value_array)
	Y_s_value_array=fillna_with_0(Y_s_value_array)
	Z_s_value_array=fillna_with_0(Z_s_value_array)
	S_combine_array=[]
	for (x,y,z) in zip(X_s_value_array,Y_s_value_array,Z_s_value_array):
		S_combine_array.append(distance_combine(x,y,z))
	return S_combine_array


per_scene_amount_of_movement_list=[]
for sub_acc_index,sub_acc_array in enumerate(sub_acc_array_list):
	X_s_value_array=[]
	Y_s_value_array=[]
	Z_s_value_array=[]
	for i in range(1, int(len(sub_acc_array) / 50) + 1):
		sub_df = sub_acc_array.iloc[50 * (i - 1):50 * i, :]
		X_a_value=sub_df.iloc[:,0]
		Y_a_value=sub_df.iloc[:,1]
		Z_a_value=sub_df.iloc[:,2]
		X_s_value_array.append(acc2S_value(X_a_value))
		Y_s_value_array.append(acc2S_value(Y_a_value))
		Z_s_value_array.append(acc2S_value(Z_a_value))
	'''
	plt.figure(12,figsize=(16,9), facecolor = 'gray')
	plt.subplot(131)
	plt.xlabel("time")
	plt.ylabel("X_s_value")
	plt.title("X_s_value")
	plt.bar(range(len(X_s_value_array)), X_s_value_array,fc='r')
	plt.subplot(132)
	plt.xlabel("time")
	plt.ylabel("Y_s_value")
	plt.title("Y_s_value")
	plt.bar(range(len(Y_s_value_array)), Y_s_value_array,fc='y')
	plt.subplot(133)
	plt.xlabel("time")
	plt.ylabel("Z_s_value")
	plt.title("Z_s_value")
	plt.bar(range(len(Z_s_value_array)), Z_s_value_array,fc='g')
	plt.show()
	plt.close()
	'''
	S_combine_arrays=construct_S(X_s_value_array, Y_s_value_array, Z_s_value_array)
	'''
	plt.figure()
	plt.bar(range(1,len(S_combine_arrays)+1),S_combine_arrays)
	plt.xticks(range(1,len(S_combine_arrays)+1))
	plt.xlabel("seconds")
	plt.ylabel("amount of movement")
	plt.title("The amount of movement per minute")
	plt.show()
	plt.close()
	'''
	print("the subarray index is ",sub_acc_index+1," and its total amount of movement are :",round(sum(S_combine_arrays),3))
	per_scene_amount_of_movement_list.append(round(sum(S_combine_arrays),3))

#print(per_scene_amount_of_movement_list)
plt.figure()
plt.bar(range(1,len(per_scene_amount_of_movement_list)+1), per_scene_amount_of_movement_list,fc='r')
plt.xticks(range(1,len(per_scene_amount_of_movement_list)+1))
plt.xlabel("scene_index")
plt.ylabel("amount_of_movement")
plt.title("per_scene_movement")
plt.savefig(per_scene_movement_png)
#plt.show()
plt.close()

scene_df['movement']=pd.Series(per_scene_amount_of_movement_list)
scene_df.to_csv(per_scene_movement_txt,columns=['sample_id','scene','movement'],sep="\t",index=False)
total_result=sensor_time_file[:-4]+"\ttotal_sensor:"+"\t"+str(round(sum(per_scene_amount_of_movement_list),3))
my_open = open(per_scene_movement_txt, 'a')
my_open.write(total_result)
my_open.close()

