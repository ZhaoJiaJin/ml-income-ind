#!/usr/bin/env python


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model, preprocessing
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFE
import sys
from sklearn.model_selection import train_test_split

# Load the diabetes dataset
#alldata = np.loadtxt("train.csv",delimiter=",", usecols=range(0,2))
#alldata = pd.read_csv("train.csv").to_numpy()
#predictdata = pd.read_csv("predict.csv").to_numpy()
alldata = pd.read_csv("train.csv")
predictdata = pd.read_csv("predict.csv")



imp = SimpleImputer(strategy='most_frequent')
alldata = imp.fit_transform(alldata)
predictdata = imp.fit_transform(predictdata)
(ROW,COL) = alldata.shape
(PROW,PCOL) = predictdata.shape



#le = preprocessing.LabelEncoder()
#
#
#





X = alldata[:, 1:11]
Y = alldata[:, 11]


predictX = predictdata[:,1:11]

newX = np.zeros((ROW,0))
newPX = np.zeros((PROW,0))
strlist = []
for i in range(0,10):
    print(i, type(X[:,i][0]))
    print(i,X[:5,i])
    if type(X[:,i][0]) == str:
        strlist.append(i)
    else:
        newX = np.append(newX, X[:, i:i+1], axis=1)
        newPX = np.append(newPX, predictX[:, i:i+1], axis = 1)

print(strlist)

labeldata = X[:,strlist]
labeldata = np.append(labeldata, predictX[:,strlist],axis=0)


print(labeldata.shape)
print(labeldata[:2,:])

print("one hot encoder")
for i in range(0,len(strlist)):
    feaLen = len(np.unique(labeldata[:,i:i+1]))
    ldata = labeldata[:,i:i+1]
    if feaLen < 10:
        drop_enc = preprocessing.OneHotEncoder(drop='first',categories='auto').fit(ldata)
        newdata = drop_enc.transform(ldata).toarray()
        newX = np.append(newX, newdata[:ROW, :], axis=1)
        newPX = np.append(newPX, newdata[ROW:,:], axis = 1)
    else:
        enc = preprocessing.OrdinalEncoder().fit(ldata)
        newdata = enc.transform(ldata)
        newX = np.append(newX, newdata[:ROW, :], axis=1)
        newPX = np.append(newPX, newdata[ROW:,:], axis = 1)



#drop_enc = preprocessing.OneHotEncoder(drop='first',categories='auto').fit(labeldata)
#newdata = drop_enc.transform(labeldata).toarray()
#newX = np.append(newX, newdata[:ROW, :], axis=1)
#newPX = np.append(newPX, newdata[ROW:,:], axis = 1)

X = newX
predictX = newPX

y = Y




print("standard scaler")
scaler = preprocessing.StandardScaler().fit(X)
X = scaler.transform(X)
predictX = scaler.transform(predictX)

(XROW,XCOL) = X.shape

#no of features
nof_list=np.arange(1,XCOL)
high_score=0
#Variable to store the optimum features
nof=0
score_list =[]
choosemodel = None
cols = None
print("x.shape:",X.shape)
print("nof_list:",len(nof_list))
for n in range(len(nof_list)):
    print("RFE:", n)
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)
    model = linear_model.LinearRegression()
    rfe = RFE(model,nof_list[n])
    X_train_rfe = rfe.fit_transform(X_train,y_train)
    X_test_rfe = rfe.transform(X_test)
    model.fit(X_train_rfe,y_train)
    score = model.score(X_test_rfe,y_test)
    score_list.append(score)
    if(score>high_score):
        high_score = score
        nof = nof_list[n]
        choosemodel = model
        cols = rfe.support_
print("Optimum number of features: %d" %nof)
print("Score with %d features: %f" % (nof, high_score))



#model = linear_model.LinearRegression()
##Initializing RFE model
#rfe = RFE(model, 7)
##Transforming data using RFE
#X_rfe = rfe.fit_transform(X,y)
##Fitting the data to model
#model.fit(X_rfe,y)
#cols = rfe.support_
#
predictX = predictX[:, cols]
predictRes = choosemodel.predict(predictX)

#sys.exit(9)
## Plot outputs
#plt.scatter(testX, testY,  color='black')
#plt.plot(testX, predictY, color='blue', linewidth=3)
#
#plt.xticks(())
#plt.yticks(())
#
#plt.show()

with open("result.csv","w") as f:
    index = predictdata[:, 0]
    f.write("Instance,Income\n")
    (pre_data_row,pre_data_col) = predictdata.shape
    for i in range(0, pre_data_row):
        f.write("%d,%d\n" % (index[i], predictRes[i]))
