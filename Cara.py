from numpy import genfromtxt
import numpy as np
from sklearn import datasets

dataPath = "NVDA_ORG.csv"
nvdaDB = genfromtxt(dataPath, delimiter=',')
X = nvdaDB[:, :-1]
Y = nvdaDB[:, -1]
print(X[1][1])