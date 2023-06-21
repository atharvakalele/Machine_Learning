# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 11:23:57 2023

@author: athar
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path = r"C:\Users\athar\OneDrive\Desktop\Machine learning\Projects\SimpleLinearRegressionDataset\HtWt.csv"

df = pd.read_csv(path)

X = df['Height']
y = df['Weight']


#explore the dataset
plt.scatter(X, y, color='blue')
plt.xlabel('Height')
plt.ylabel('Weight')
plt.show()

#data preprocessing
X = np.asanyarray(X).reshape(-1, 1)
y = np.asanyarray(y)

#splitting the dataset
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)

#training
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

slope = regressor.coef_
intercept = regressor.intercept_

#visualizing the model
equation = slope*X_test + intercept

plt.scatter(X_test, y_test, color = 'blue')
plt.plot(X_test, equation,color = 'green')
plt.show()

#evaluating the model
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, equation)
r_squared = r2_score(y_test, equation)
print('Mean_Squared_Error :' ,mse)
print('r_square_value :',r_squared)

