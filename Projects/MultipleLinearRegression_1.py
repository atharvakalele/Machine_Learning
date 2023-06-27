# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 13:56:47 2023

@author: athar
"""

import pandas as pd
import numpy as np

path = r"C:\Users\athar\OneDrive\Desktop\Machine learning\Projects\MultipleLinearRegressionDataset\50_Startups.csv"

data = pd.read_csv(path)

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder = LabelEncoder()
X[:,3] = labelencoder.fit_transform(X[:,3])

onehotencoder = OneHotEncoder(sparse=False)
column3encoded = onehotencoder.fit_transform(X[:,3].reshape(-1,1))
X = np.concatenate((column3encoded,X[:,:3]),axis=1)
X = X[:,0:5]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

#Backward Elimination
import statsmodels.api as sm

X = np.append(np.ones((50, 1)).astype(int), X, axis=1)

def backward_elimination(x,y,sl):
    num_var = x.shape[1]
    temp = np.zeros((x.shape[0],num_var)).astype(int)
    for i in range(num_var): #iterates over each variable in the columns.
        regressor_OLS = sm.OLS(endog=y, exog = x).fit()
        max_p_value = max(regressor_OLS.pvalues).astype(float)
        #check for p_value in the first pass
        
        if max_p_value < sl:
            break
        for j in range(num_var-i): #assuming after the 1st iteration we have remove the ith time variable.
            if max_p_value == regressor_OLS.pvalues[j].astype(float):
                temp[:,j] = x[:,j]
                x = np.delete(x, j, 1)
                break
    return x

X_opt = backward_elimination(X.astype(float), y.astype(float), sl = 0.01).astype(float)




