# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 12:56:38 2023

@author: athar
"""

import pandas as pd
import numpy as np
import seaborn as sns

df = pd.read_csv(r"C:\Users\athar\OneDrive\Desktop\Machine learning\Projects\MultipleLinearRegressionDataset\multiple_linear_regression_dataset.csv")
X = df.iloc[:, [0, 1]].values
y = df.iloc[:, -1].values

# Checking linear regression assumptions.
# 1 Linear dependency
sns.relplot(data=df, x='age', y='income', kind='scatter')
sns.relplot(data=df, x='experience', y='income', kind='scatter')

# Dropping the age column if X has more than one column
if X.shape[1] > 1:
    X = np.delete(X, 0, 1)

# Adding a column of ones as the first column for the intercept term
X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)

# 2 Multicollinearity
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = []
for i in range(X.shape[1]):
    vif.append(variance_inflation_factor(X, i))

#train the model
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state = 42)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,y_train)

y_pred = lr.predict(X_test)
#3 Normal residual
residual = y_pred - y_test

sns.displot(residual,kind='kde')

from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

    
