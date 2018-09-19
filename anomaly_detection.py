from __future__ import division
import argparse
import matplotlib.pyplot as plt
from numpy import linspace, loadtxt, ones, convolve
import numpy as np
import pandas as pd
import collections
from random import randint
from matplotlib import style
from itertools import  count
import glob
import os

def parse_arg():
	parser = argparse.ArgumentParser(description = "Description for my parser")
	parser.add_argument("--path", required = True)
	argument = parser.parse_args()
	if argument.path:
		data_folder_path = argument.path
	return data_folder_path


def get_all_raw_files(input):      
	path = input+'/*csv' 
	file_map = []   
	files=glob.glob(path)
	data_frame = None
	for file in files:     
	  	file_map.append(file)
	return file_map


def read_data_from_files(file_map):
	data_frame = None
	for file in file_map:     
		df = pd.read_csv(file, header=None)
		data_frame = pd.concat([data_frame,df], axis=0)
	return data_frame


def moving_average(data, window_size):
	window = np.ones(int(window_size))/float(window_size)
	return np.convolve(data, window, 'same')


def detect_anomalies(y, window_size, sigma=1.0):

	average = moving_average(y, window_size).tolist()
	residual = y - average
	# Calculate the variation in the distribution of the residual
	std = np.std(residual)
	print ("STD: {}".format(std))
	order_dict = collections.OrderedDict()
	for index, val, avg_val in zip(count(), y, average):
		if (np.abs(val-avg_val)) >= (sigma*std):
			order_dict[index] = val
	return order_dict

def plot_results(x, y, window_size, sigma=1):
  
	plt.figure(figsize=(20, 15))
	plt.plot(x, y, "g.")
	y_av = moving_average(y, window_size)
	plt.plot(x, y_av, color='black')
	plt.xlim(0, len(y))
	ordered_dict = detect_anomalies(y,window_size, sigma)
	x_anomaly = np.fromiter(ordered_dict.keys(), dtype=int, count=len(ordered_dict))
	y_anomaly = np.fromiter(ordered_dict.values(), dtype=float,count=len(ordered_dict))

	plt.plot(x_anomaly, y_anomaly, "r*", markersize=10)
	plt.grid(True)
	plt.xlabel('Time')
	plt.ylabel('Clicks')
	plt.title("Anomalies in Timeseries data")
	plt.show()

def remove_noise_from_anomaly_dict(anomalies_dict):
	#Remove noise
	# Less than five Contiguous anomalies are considered noise here
	i =0
	arr = list(anomalies_dict.keys())
	index = -1
	final_anomalies = []
	for i in range(len(arr)-1):
		if arr[i+1] == arr[i] +1:
			index = i
			while i < len(arr)-1 and (arr[i+1] == arr[i] +1):
				i = i+1
        # if found more than 5 contiguous anomalous points
		if i -index>5:
			j = index
			while j< i:
				final_anomalies.append(arr[j])
				j = j+1
	return final_anomalies


def main():
	# Window Size : 1 hour => last 60 data points
	window_size = 60
	# Threshold for anomalous datapoint : if abs(datapoint -avg) > std* thresold then anomalous
	sigma = 4
	# Get data folder path
	data_folder_path = parse_arg()
	# Get all files in data folder
	file_map = get_all_raw_files(data_folder_path)
	# Build all data
	data_frame = read_data_from_files(file_map)
	
	# Get data from pandas dataframe
	clicks = data_frame[1].values
	timestamp = data_frame[0].values

	X = np.asarray([i for  i in range(len(clicks))])
	# Get timestamp and anomaly
	anomalies_dict = detect_anomalies(clicks,window_size, sigma)

	# Plot data and anomaly
	plot_results(X, y=clicks, window_size=window_size, sigma=sigma)
	
	# After removing noise, this is the list of anomaly indicies
	final_anomalies_indices = remove_noise_from_anomaly_dict (anomalies_dict)

	minutes_in_day = 60*24
	
	#initialize the empty list of list
	anomaly_per_day =[[] for i in range (len(file_map))]
	for i in final_anomalies_indices:
		index = (int)(i/minutes_in_day)
		anomaly_per_day[index].append("Timestamp:{}, Value:{}".format(timestamp[i],anomalies_dict[i]))

    # Print anomay data per day
	for i in range(len(anomaly_per_day)):
		if len(anomaly_per_day[i]) == 0:
			print ("{} : No anomalies".format(file_map[i]))
		else:
			print ("{} : {}".format(file_map[i], anomaly_per_day[i]))



if __name__ == '__main__':
    main()