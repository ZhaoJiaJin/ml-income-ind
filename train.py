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
from sklearn.ensemble import RandomForestRegressor

# Load the diabetes dataset
#alldata = np.loadtxt("train.csv",delimiter=",", usecols=range(0,2))
alldata = pd.read_csv("train.csv").to_numpy()
predictdata = pd.read_csv("predict.csv").to_numpy()
#alldata = pd.read_csv("train.csv")
#predictdata = pd.read_csv("predict.csv")



#imp = SimpleImputer(strategy='most_frequent')
#alldata = imp.fit_transform(alldata)
#predictdata = imp.fit_transform(predictdata)
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
    if type(X[:,i][0]) == str:
        strlist.append(i)
    else:
        imp = SimpleImputer(strategy='median')
        newX = np.append(newX, imp.fit_transform(X[:, i:i+1]), axis=1)
        newPX = np.append(newPX, imp.fit_transform(predictX[:, i:i+1]), axis = 1)



scaler = preprocessing.StandardScaler().fit(newX)
newX = scaler.transform(newX)
newPX = scaler.transform(newPX)
newX = preprocessing.normalize(newX, norm='l2')
newPX = preprocessing.normalize(newPX, norm='l2')



labeldata = X[:,strlist]
labeldata = np.append(labeldata, predictX[:,strlist],axis=0)



print("one hot encoder")
for i in range(0,len(strlist)):
    #print(labeldata[:,i:i+1])
    ldata = labeldata[:,i:i+1]
    imp = SimpleImputer(strategy='most_frequent')
    ldata = imp.fit_transform(ldata)
    feaLen = len(np.unique(ldata))
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




X = newX
predictX = newPX

y = Y




#print("standard scaler")
#scaler = preprocessing.StandardScaler().fit(X)
#X = scaler.transform(X)
#predictX = scaler.transform(predictX)

(XROW,XCOL) = X.shape



#train_features, test_features, train_labels, test_labels = train_test_split(X, y, test_size = 0.2, random_state = 42)
train_features = X
train_labels = y

print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)

# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
# Train the model on training data
rf.fit(train_features, train_labels);

predictRes = rf.predict(predictX)

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
