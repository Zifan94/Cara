from numpy import *
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LinearRegression
import csv
import datetime
import time
import matplotlib.pyplot as plt
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
        col = int(Date[i].month)
        X[i-30][32+col] = 1.0

    return X
            

if __name__ == "__main__":
    
    # dataPath = "Data/NVDA_TEST.csv"
    dataPath = "Data/NVDA_ORG.csv"
    nvdaDB = genfromtxt(dataPath, delimiter=',')

    all_history_price = nvdaDB[0:, 4]
    sampleCount = nvdaDB.shape[0]
    
    train_X = zeros((sampleCount-30, 45))
    train_Y = calc_Y(sampleCount, nvdaDB)
    train_X += calc_0_to_29(sampleCount, all_history_price)
    
     
    erdatePath='Data/NVDA_ER_Date.csv'
    ERdate = get_Date(erdatePath)
    date = get_Date(dataPath)
    train_X += calc_30_to_32(sampleCount, ERdate,date)

    train_X += calc_33_to_44(sampleCount, date)
    # get rid of first 400 example since its too long
    train_X = train_X[400:, ]
    train_Y = train_Y[400:, ]

    # Train_X, Train_Y constructed
    np.savetxt("OUT-train_X.csv", train_X, fmt='%s', delimiter=",")
    np.savetxt("OUT-train_Y.csv", train_Y, fmt='%s', delimiter=",")

    print("training example count: "+str(nvdaDB.shape[0]-30-400))
    print("training data X: "+str(train_X.shape))
    print("training data Y: "+str(train_Y.shape))
    assert nvdaDB.shape[0]-30-400 == train_X.shape[0]
    assert nvdaDB.shape[0]-30-400 == train_Y.shape[0]

    # Using Linear Regression
    date_X = zeros((nvdaDB.shape[0]-30-400-1-500, 1))
    date_cnt = 1
    test_Y = zeros((nvdaDB.shape[0]-30-400-1-500, 1))
    pred_Y = zeros((nvdaDB.shape[0]-30-400-1-500, 1))
    for i in range(500, nvdaDB.shape[0]-30-400-1):
        tmp_train_X = train_X[i:,]
        tmp_train_Y = train_Y[i:,]
        linearReg = LinearRegression()
        linearReg.fit(tmp_train_X, tmp_train_Y)
        tmp_test_X = train_X[i+1,]
        tmp_test_Y = train_Y[i+1][0]
        pred_Y[i-500][0] = (linearReg.predict(tmp_test_X.reshape(1, -1))[0][0])
        test_Y[i-500][0] = (tmp_test_Y)
        date_X[i-500][0] = date_cnt
        date_cnt += 1

    sum = 0
    for i in range(0, test_Y.shape[0]):
        dis = test_Y[i][0] - pred_Y[i][0]
        disSq = dis * dis
        sum += disSq
    MSE = sum / test_Y.shape[0]
    print("")
    print("Min Squared Error: "+str(round(sqrt(MSE), 2)))
    print("")

    print("dateCnt data X: "+str(date_X.shape))
    print("testing data Y: "+str(test_Y.shape))
    print("predict data Y: "+str(pred_Y.shape))


    # Plot
    fig, ax = plt.subplots()
    plt.plot(date_X, test_Y, 'r', label = "target", lw=2)
    plt.plot(date_X, pred_Y, 'b', label = "predict", lw=1)
    plt.legend(loc = 0)
    plt.show()

