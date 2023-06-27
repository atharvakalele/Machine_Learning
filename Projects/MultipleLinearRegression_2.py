# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 23:59:46 2023

@author: athar
"""

import numpy as np
import pandas as pd

dataset = pd.read_csv(
    r"C:\Users\athar\OneDrive\Desktop\Machine learning\Projects\MultipleLinearRegressionDataset\50_Startups.csv")

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

#identify categorical column
idx= -1
for i in range(len(X[0])):
    if type(X[0,i]) == str:
        idx = i
        break

    
#one-hot-encoding(assuming the categorical column is last)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:,idx] = labelencoder.fit_transform(X[:,idx])
onehotencoder = OneHotEncoder(sparse=False)
columnencoded = onehotencoder.fit_transform(X[:,idx].reshape(-1,1))
X = np.concatenate((columnencoded,X[:,:idx]),axis=1)

#train(0.8)-test(0.2)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#training the model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#evaluating the model --- 1st phase
y_pred = regressor.predict(X_test)
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
r_squared = r2_score(y_test, y_pred)
print('Mean_Squared_Error :' ,mse)
print('r_square_value :',r_squared)

#Backward-Elimination(without adj-rsqrd)
import statsmodels.api as sm

#adding constant
X = np.append(np.ones((50, 1)).astype(int), X, axis=1)

def Backward_Elimination(x,y,sl):
    num_var = x.shape[1]
    temp = np.zeros((x.shape[0],num_var)).astype(float)
    for i in range(num_var):
        regressor_OLS = sm.OLS(endog=y, exog=x).fit()
        max_pvalue = max(regressor_OLS.pvalues).astype(float)
        if max_pvalue < sl:
            break
        for j in range(num_var-i):
            if max_pvalue == regressor_OLS.pvalues[j].astype(float):
                temp[:,j] = x[:,j]
                x = np.delete(x, j,1)
                break
    return x

X_opt = Backward_Elimination(X.astype(float), y.astype(float), sl=0.05)

#train-test split --- 2nd phase
X_train,X_test,y_train,y_test = train_test_split(X_opt,y,test_size=0.2,random_state=42)

#retrain
regressor_2 = LinearRegression()
regressor_2.fit(X_train,y_train)
#evaluating the model --- 1st phase
y_pred = regressor_2.predict(X_test)
from sklearn.metrics import mean_squared_error, r2_score
mse_2 = mean_squared_error(y_test, y_pred)
r_squared_2 = r2_score(y_test, y_pred)
print('Mean_Squared_Error :' ,mse_2)
print('r_square_value :',r_squared_2)