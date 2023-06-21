# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 15:00:22 2023

@author: athar
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv(r"C:\Users\athar\OneDrive\Desktop\Machine learning\Projects\SimpleLinearRegressionDataset\Salary_Data.csv")

X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=1/3, random_state=42)


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)

#Visualizing the training set
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train),color = 'green')
plt.title("Salary Vs Experience")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()

#Visulaizing the test set
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train),color = 'green')
plt.title("Salary Vs Experience")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()

