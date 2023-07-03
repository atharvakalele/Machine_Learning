# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 02:07:20 2023

@author: athar
"""

import numpy as np
import pandas as pd

dataset = pd.read_csv(r"C:\Users\athar\OneDrive\Desktop\Machine learning\Projects\MultipleLinearRegressionDataset\50_Startups.csv")

X = dataset.drop(['Profit'],1).values
y = dataset.iloc[:,-1].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:,3] = labelencoder.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(sparse=False)
columnencoded = onehotencoder.fit_transform(X[:,3].reshape(-1,1))
X = np.concatenate((columnencoded,X[:,:3]),axis=1)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

from sklearn.metrics import mean_squared_error,r2_score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

import statsmodels.api as sm

def backward_elimination(x, y, sl):
    num_var = x.shape[1]
    temp = np.zeros((x.shape[0], num_var)).astype(float)
    
    for i in range(num_var):
        lr_ols = sm.OLS(endog=y, exog=x).fit()
        max_var = max(lr_ols.pvalues).astype(float)
        r_adj_before = lr_ols.rsquared_adj.astype(float)
        print(lr_ols.summary())
        if max_var > sl:
            for j in range(num_var-i):
                if j < x.shape[1] and max_var == lr_ols.pvalues[j].astype(float):
                    temp[:, j] = x[:, j]
                    x = np.delete(x, j, 1)
                    lr_ols = sm.OLS(endog=y, exog=x).fit()
                    r_adj_after = lr_ols.rsquared_adj.astype(float)
                    
                    if r_adj_before >= r_adj_after:
                        temp_reshaped = temp[:, 0].reshape(-1, 1)
                        x = np.concatenate((temp_reshaped, x), axis=1)
                        print(lr_ols.summary())
                        return x
                    else:
                        continue
    print(lr_ols.summary())
    return x

X_opt = backward_elimination(X.astype(float), y.astype(float), sl=0.05)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X_opt,y,test_size=0.2,random_state=42)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

from sklearn.metrics import mean_squared_error,r2_score
mse2 = mean_squared_error(y_test, y_pred)
r2_n = r2_score(y_test, y_pred)