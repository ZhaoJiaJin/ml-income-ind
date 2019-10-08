#!/usr/bin/env python


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model, preprocessing
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import sys

# Load the diabetes dataset
#alldata = np.loadtxt("train.csv",delimiter=",", usecols=range(0,2))
alldata = pd.read_csv("train.csv")
predictdata = pd.read_csv("predict.csv")

imp = SimpleImputer(strategy='most_frequent')
alldata = imp.fit_transform(alldata)
predictdata = imp.fit_transform(predictdata)
(ROW,COL) = alldata.shape

le = preprocessing.LabelEncoder()



for i in range(0,COL):
    if type(alldata[:,i][0]) == str:
        le.fit(np.append(alldata[:,i],predictdata[:, i]))
        alldata[:, i] = le.transform(alldata[:,i])
        predictdata[:, i] = le.transform(predictdata[:, i])



X = alldata[:, 1:11]
Y = alldata[:, 11]

predictX = predictdata[:,1:11]

#TODO
# Use only one feature
#diabetes_X = diabetes.data[:, np.newaxis, 2]

# Split the data into training/testing sets
#diabetes_X_train = diabetes_X[:-20]
#diabetes_X_test = diabetes_X[-20:]

# Split the targets into training/testing sets
#diabetes_y_train = diabetes.target[:-20]
#diabetes_y_test = diabetes.target[-20:]
splitp = int(0.9 * ROW)

trainX = X[:splitp, :]
testX = X[splitp:,:]

trainY = Y[:splitp]
testY = Y[splitp:]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(trainX, trainY)

# Make predictions using the testing set
predictY = regr.predict(testX)



# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(testY, predictY))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(testY, predictY))



predictRes = regr.predict(predictX)

#sys.exit(9)
## Plot outputs
#plt.scatter(testX, testY,  color='black')
#plt.plot(testX, predictY, color='blue', linewidth=3)
#
#plt.xticks(())
#plt.yticks(())
#
#plt.show()

index = predictdata[:, 0]
print("Instance,Income")
(pre_data_row,pre_data_col) = predictdata.shape
for i in range(0, pre_data_row):
    print("%d,%d" % (index[i], predictRes[i]))
