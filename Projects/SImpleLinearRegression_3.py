# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 22:53:26 2023

@author: athar
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing the dataset
path = r"C:\Users\athar\OneDrive\Desktop\Machine learning\Projects\SimpleLinearRegressionDataset\tvmarketing.csv"
df = pd.read_csv(path)

#visualizing the dataset
X = df['TV']
y = df['Sales']

plt.scatter(X, y,color='blue')
plt.show()

#Splitting the dataset
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)

X_train = X_train[:, np.newaxis]
X_test = X_test[:, np.newaxis]

#Training the model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

print(regressor.coef_, regressor.intercept_)
y_pred = regressor.predict(X_test)

#visualizing the model
plt.scatter(X_train,y_train,color = 'blue')
plt.plot(X_test,y_pred,color = 'red')

#evaluating the model
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
r_squared = r2_score(y_test, y_pred)
print('Mean_Squared_Error :' ,mse)
print('r_square_value :',r_squared)

