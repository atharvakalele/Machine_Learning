# -*- coding: utf-8 -*-
"""
Created on Sat Jul  1 12:21:56 2023

@author: athar
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

dataset = pd.read_csv(r"C:\Users\athar\OneDrive\Desktop\Machine learning\Projects\MultipleLinearRegressionDataset\kc_house_data.csv")

dataset = dataset.drop(['id','date'], axis = 1)

"""
with sns.plotting_context("paper", font_scale=4):
    g = sns.pairplot(dataset[['sqft_lot','sqft_above','price','sqft_living','bedrooms']], 
                     hue='bedrooms', palette='tab20', size=9)
    g.set(xticklabels=[])
"""

X = dataset.iloc[:,1:].values
y = dataset.iloc[:,0].values



#training the model->1st Phase
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#evaluation->1st Phase
y_pred = regressor.predict(X_test)

from sklearn import metrics

r2_score = metrics.r2_score(y_test, y_pred)
print("R2 Score:", r2_score)

mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)

#backward elimination->phase 2

import statsmodels.api as sm

def backward_elimination(x,y,sl):
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

#training -> phase 2
X_opt = backward_elimination(X.astype(float), y.astype(float), sl=0.01)

#split
X_opt_train,X_opt_test,y_train,y_test = train_test_split(X_opt,y,test_size=0.2,random_state=42)


regressor_2 = LinearRegression()
regressor_2.fit(X_opt_train,y_train)

y_pred_2 = regressor_2.predict(X_opt_test)

r2_score = metrics.r2_score(y_test, y_pred_2)
print("R2 Score:", r2_score)

mse = metrics.mean_squared_error(y_test, y_pred_2)
rmse = np.sqrt(mse)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)