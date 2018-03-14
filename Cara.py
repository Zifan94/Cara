from numpy import *
import numpy as np
from sklearn import datasets
import csv
import datetime
import time
def calc_Y(sampleCount, nvdaDB):    # process the close_price into Y
    Y = zeros((sampleCount-30, 1))
    for i in range(30, sampleCount):
        Y[i-30][0] = nvdaDB[i][4]  
    return Y


def calc_0_to_29(sampleCount, all_history_price):    # [0] to [29]  price_day-1 to price_day-30
    X = zeros((sampleCount-30, 45))
    for i in range(30, sampleCount):
        for j in range(0, 30):
            X[i-30][j] = round(all_history_price[i-j-1], 2)
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
    


def calc_30_to_32(sampleCount, ER_Date, Date):
    X= zeros((sampleCount-30, 45))
    for i in range(30, sampleCount):
        for j in range(0, len(ER_Date)-1):
            if Date[i] <= ER_Date[j+1]:
                if Date[i] > ER_Date[j]:
                    X[i-30][32] = (Date[i] - ER_Date[j]).days
                    X[i-30][31] = (ER_Date[j+1] - Date[i]).days
                    if X[i-30][31]==0 or X[i-30][32]==0:
                        X[i-30][30] = 1
                else:
                    X[i-30][31] = (ER_Date[j] - Date[i]).days
#                    print(X[i-30][1])
                    X[i-30][32] = 9999
                    if X[i-30][31]==0:
                        X[i-30][30] = 1
                break
#        print(str(X[i-29][0])+"\t"+str(X[i-29][1])+"\t"+str(X[i-29][2]))
    return X
                

def calc_33_to_44(sampleCount, Date):
    X= zeros((sampleCount-30, 45))
    for i in range(30, sampleCount):
        # m_d_y = Date[i].split('-')
        # col = m_d_y[0]
        col = int(Date[i].month)
        print(col)
        X[i-30][32+col] = 1.0

    return X
            

if __name__ == "__main__":
    
    dataPath = "Data/NVDA_TEST.csv"
    # dataPath = "Data/NVDA_ORG.csv"
    nvdaDB = genfromtxt(dataPath, delimiter=',')

    all_history_price = nvdaDB[0:, 4]
    sampleCount = nvdaDB.shape[0]
    
    result_X = zeros((sampleCount-30, 45))
    Y = calc_Y(sampleCount, nvdaDB)
    result_X += calc_0_to_29(sampleCount, all_history_price)
    
     
    erdatePath='Data/NVDA_ER_Date.csv'
    ERdate = get_Date(erdatePath)
    print(ERdate[0+1])
    date = get_Date(dataPath)
    result_X += calc_30_to_32(sampleCount, ERdate,date)

    result_X += calc_33_to_44(sampleCount, date)

    print(datetime.datetime.strptime("11-6-2014", "%m-%d-%Y") - datetime.datetime.strptime("4-15-2013", "%m-%d-%Y"))
    np.savetxt("OUT-result.csv", result_X, fmt='%s', delimiter=",")
