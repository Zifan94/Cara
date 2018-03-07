from numpy import *
import numpy as np
from sklearn import datasets
import csv
import datetime
import time
def calc_Y(sampleCount, nvdaDB):	# process the close_price into Y
	Y = zeros((sampleCount-30, 1))
	for i in range(30, sampleCount):
		Y[i-30][0] = nvdaDB[i][4]  
	return Y


def calc_0_to_29(sampleCount, all_history_price):	# [0] to [29]  price_day-1 to price_day-30
	X = zeros((sampleCount-30, 45))
	for i in range(29, sampleCount-1):
		for j in range(0, 29):
			X[i-29][j] = all_history_price[i-j-1]
	return X

def get_Date(datapath):                 #read date
    with open(datapath,'r') as csvfile:
        reader = csv.reader(csvfile)
        Date = [row[0] for row in reader]
    for i in range(0,len(Date)):
        Date[i]=Date[i].replace("/","-")
        Date[i]=datetime.datetime.strptime(Date[i], "%m-%d-%Y")
#    print(Date[0])
    return Date
    


def calc_30_to_32(ER_Date,Date):
    X= zeros((sampleCount-30, 45))
    for i in range(29, sampleCount-1):
        for j in range(0, len(ER_Date)-1):
            if Date[i-29] <= ER_Date[j+1]:
                if Date[i-29] > ER_Date[j]:
                    X[i-29][32] = (Date[i-29] - ER_Date[j]).days
                    X[i-29][31] = (ER_Date[j+1] - Date[i-29]).days
                    if X[i-29][31]==0 or X[i-29][32]==0:
                        X[i-29][30] = 1
                else:
                    X[i-29][31] = (ER_Date[j] - Date[i-29]).days
#                    print(X[i-30][1])
                    X[i-29][32] = 9999
                    if X[i-29][31]==0:
                        X[i-29][30] = 1
                break
#        print(str(X[i-29][0])+"\t"+str(X[i-29][1])+"\t"+str(X[i-29][2]))
    return X
                
            

if __name__ == "__main__":
	
#    dataPath = "Data/NVDA_TEST.csv"
    dataPath = "Data/NVDA_ORG.csv"
    nvdaDB = genfromtxt(dataPath, delimiter=',')

    all_history_price = nvdaDB[1:, 4]
    sampleCount = nvdaDB.shape[0]
	
    result_X = zeros((sampleCount-30, 45))
    Y = calc_Y(sampleCount, nvdaDB)
    result_X += calc_0_to_29(sampleCount, all_history_price)
    
     
    erdatePath='Data/NVDA_ER_Date.csv'
    ERdate = get_Date(erdatePath)
#    print(ERdate[0+1])
    date = get_Date(dataPath)
    result_X += calc_30_to_32(ERdate,date)
#    print(date[1])
