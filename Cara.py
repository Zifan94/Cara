from numpy import *
import numpy as np
from sklearn import datasets

def calc_Y(sampleCount, nvdaDB):	# process the close_price into Y
	Y = zeros((sampleCount-31, 1))
	for i in range(31, sampleCount):
		Y[i-31][0] = nvdaDB[i][4]  
	return Y


def calc_0_to_29(sampleCount, all_history_price):	# [0] to [29]  price_day-1 to price_day-30
	X = zeros((sampleCount-31, 45))
	for i in range(30, sampleCount-1):
		for j in range(0, 30):
			X[i-30][j] = all_history_price[i-j-1]
	return X


if __name__ == "__main__":
	
	dataPath = "Data/NVDA_TEST.csv"
	nvdaDB = genfromtxt(dataPath, delimiter=',')

	all_history_price = nvdaDB[1:, 4]
	sampleCount = nvdaDB.shape[0]
	
	result_X = zeros((sampleCount-31, 45))
	Y = calc_Y(sampleCount, nvdaDB)
	result_X += calc_0_to_29(sampleCount, all_history_price)
