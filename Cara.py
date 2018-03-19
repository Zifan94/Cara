from numpy import *
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LinearRegression
import csv
import datetime
import time
import matplotlib.pyplot as plt
from sklearn.svm import SVC

def red(text):
    return "\033[1;31m{}\033[0;0m".format(text)
def blue(text):
    return "\033[1;34m{}\033[0;0m".format(text)
def green(text):
    return "\033[1;32m{}\033[0;0m".format(text)

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
    pred_date =None
    with open(datapath,'r') as csvfile:
        reader = csv.reader(csvfile)
        Date = [row[0] for row in reader]
    for i in range(0,len(Date)):
        Date[i]=Date[i].replace("/","-")
        Date[i]=datetime.datetime.strptime(Date[i], "%Y-%m-%d")
        pred_date = Date[i]
#    print(Date[0])
    pred_date = pred_date+datetime.timedelta(days=1)
    return Date, pred_date

def get_ER_Date(datapath):                 #read date
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
            
def show_Logo():
    print("")
    print("  CCCCCCC         AA           RRRRRRRRRR         AA    ")
    print("CC               A  A          R        R        A  A       ")
    print("C               A    A         R        R       A    A         ")
    print("C              A      A        R        R      A      A        ")
    print("C             AAAAAAAAAA       RRRRRRRRRR     AAAAAAAAAA      ")
    print("C             A        A       R   R          A        A       ")
    print("CC            A        A       R     R        A        A       ")
    print("  CCCCCCC     A        A       R       R      A        A       ")
    print("")
    print("               copy right @2018 Zifan Yang. All rights reserved.")
    print("----------------------------------------------------------------")
    print("")


def LinearRegressionEvalue():
    # Using Linear Regression
    date_X = zeros((nvdaDB.shape[0]-30-offset-1-500, 1))
    date_cnt = 1
    test_Y = zeros((nvdaDB.shape[0]-30-offset-1-500, 1))
    pred_Y = zeros((nvdaDB.shape[0]-30-offset-1-500, 1))
    for i in range(500, nvdaDB.shape[0]-30-offset-1):
        tmp_train_X = train_X[0:i,]
        tmp_train_Y = train_Y[0:i,]
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


def LinearRegressionActualPredict():
    print("")
    print(red("CARA: Predicting The Latest Price..."))
    # Actual predict
    linearReg = LinearRegression()
    linearReg.fit(train_X, train_Y)
    
    test_X = X = zeros((1, 45))
    for i in range(0, 29):
        test_X[0][i+1] = train_X[-1][i]
    test_X[0][0] = train_Y [-1][0]
    test_X[0][30] = 0
    test_X[0][31] = train_X[-1][31]-1
    test_X[0][32] = train_X[-1][32]+1
    col = int(pred_date.month)
    test_X[0][32+col] = 1.0
    ans = linearReg.predict(test_X.reshape(1, -1))[0][0]
    print("  Constructing tomorrow's input vector:")
    print("\n   "+str(test_X))
    print("\n  Predicting Result:")
    print(blue("    Date:             ")+str(pred_date))
    print(blue("    Prediction Price: ")+str(ans))
    print("")


def get_Y_cls(Y_cls):
    ret_Y = zeros((sampleCount-30, 1))
    prev = 0
    for i in range(0, Y_cls.shape[0]):
        if Y_cls[i][0] <=prev:
            ret_Y[i][0] = -1
        else:
            ret_Y[i][0] = 1
        prev = Y_cls[i][0]
    return ret_Y

def SVMEvalue():
    # Using Linear Regression
    date_X = zeros((nvdaDB.shape[0]-30-offset-1-500, 1))
    date_cnt = 1
    test_Y = zeros((nvdaDB.shape[0]-30-offset-1-500, 1))
    pred_Y = zeros((nvdaDB.shape[0]-30-offset-1-500, 1))
    for i in range(500, nvdaDB.shape[0]-30-offset-1):
        tmp_train_X = train_X[0:i,]
        tmp_train_Y = train_Y_cls[0:i,]
        svm = SVC(gamma = 'auto')
        svm.fit(tmp_train_X, tmp_train_Y.ravel())
        tmp_test_X = train_X[i+1,]
        tmp_test_Y = train_Y_cls[i+1][0]
        pred_Y[i-500][0] = (svm.predict(tmp_test_X.reshape(1, -1)))
        test_Y[i-500][0] = (tmp_test_Y)
        date_X[i-500][0] = date_cnt
        date_cnt += 1

    sumErr = 0
    for i in range(0, test_Y.shape[0]):
        if test_Y[i][0] != pred_Y[i][0]:
            sumErr += 1
    print(sumErr)
    ErrRate = double(sumErr) / double(test_Y.shape[0])
    print("")
    print("SVM Error rate on test Data: "+str(round(ErrRate, 4)*100))+"%"
    print("")

    # print("dateCnt data X: "+str(date_X.shape))
    # print("testing data Y: "+str(test_Y.shape))
    # print("predict data Y: "+str(pred_Y.shape))


    # # Plot
    # fig, ax = plt.subplots()
    # plt.plot(date_X, test_Y, 'r', label = "target", lw=2)
    # plt.plot(date_X, pred_Y, 'b', label = "predict", lw=1)
    # plt.legend(loc = 0)
    # plt.show()

if __name__ == "__main__":
    
    # dataPath = "Data/NVDA_TEST.csv"
    # dataPath = "Data/NVDA_ORG.csv"
    show_Logo()
    dataPath = "Data/NVDA.csv"
    nvdaDB = genfromtxt(dataPath, delimiter=',')

    all_history_price = nvdaDB[0:, 4]
    sampleCount = nvdaDB.shape[0]
    
    train_X = zeros((sampleCount-30, 45))
    train_Y = calc_Y(sampleCount, nvdaDB)
    train_X += calc_0_to_29(sampleCount, all_history_price)
    
     
    erdatePath='Data/NVDA_ER_Date.csv'
    ERdate = get_ER_Date(erdatePath)
    date, pred_date = get_Date(dataPath)
    train_X += calc_30_to_32(sampleCount, ERdate,date)

    train_X += calc_33_to_44(sampleCount, date)
    # get rid of first 400 example since its too long
    offset = 3977
    train_Y_cls = train_Y
    train_X = train_X[offset:, ]
    train_Y = train_Y[offset:, ]
    train_Y_cls = get_Y_cls(train_Y_cls)
    train_Y_cls = train_Y_cls[offset:, ]

    # Train_X, Train_Y constructed
    print(red("CARA: Generating Training Dataset..."))
    np.savetxt("OUT-train_X.csv", train_X, fmt='%s', delimiter=",")
    print("  \"OUT-train_X.csv\"                 "+green("saved"))
    np.savetxt("OUT-train_Y_Regression.csv", train_Y, fmt='%s', delimiter=",")
    print("  \"OUT-train_Y_Regression.csv\"      "+green("saved"))
    np.savetxt("OUT-train_Y_Classification.csv", train_Y_cls, fmt='%s', delimiter=",")
    print("  \"OUT-train_Y_Classification.csv\"  "+green("saved"))

    print("")
    print("  training sample count: "+str(nvdaDB.shape[0]-30-offset))
    print("  training data X:     "+str(train_X.shape))
    print("  training data Y reg: "+str(train_Y.shape))
    print("  training data Y cls: "+str(train_Y_cls.shape))
    assert nvdaDB.shape[0]-30-offset == train_X.shape[0]
    assert nvdaDB.shape[0]-30-offset == train_Y.shape[0]
    assert nvdaDB.shape[0]-30-offset == train_Y_cls.shape[0]
    print(blue("  Matrix Calibration ")+green("Complete"))

    # LinearRegressionEvalue()
    LinearRegressionActualPredict()
    
    # SVMEvalue()


    print("")
    print(red("CARA: All Jobs Done, Quit Now"))
    print("")