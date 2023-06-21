# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 11:38:54 2023

@author: athar
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import urllib.request

#Downloading the dataset
url = "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/FuelConsumptionCo2.csv"
filename = "FuelConsumption.csv"

urllib.request.urlretrieve(url, filename)

#import the dataset as dataframe object
path = r"C:\Users\athar\OneDrive\Desktop\Machine learning\Projects\SimpleLinearRegressionDataset\FuelConsumption.csv"
df = pd.read_csv(path)


#Explore the dataset
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
viz = cdf[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
viz.hist()
plt.show()

plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, color="blue")
plt.xlabel("ENGINESIZE")
plt.ylabel("CO2EMISSIONS")
plt.show()

plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS, color="blue")
plt.xlabel("CYLINDERS")
plt.ylabel("CO2EMISSIONS")
plt.show()

plt.scatter(cdf.FUELCONSUMPTION_COMB,cdf.CO2EMISSIONS,color='blue')
plt.xlabel("FUELCONSUMPTION_COMB")
plt.ylabel("C02EMISSIONS")
plt.show()

#splitting the dataset
msk = np.random.rand(len(df))<0.8
train = cdf[msk]
test = cdf[~msk]

#Visualizing the most related variable
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color = 'yellow')
plt.show()

#Finding the approximate parameters of the linear model using training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regressor.fit(train_x,train_y)

# Writing the linear equation
slope = regressor.coef_[0][0]
intercept = regressor.intercept_[0]
equation = train_x * slope + intercept

#visualizing the model
plt.scatter(train_x, train_y,color = 'red')
plt.plot(train_x,equation,color = 'green')
plt.show()

#evaluating the model
from sklearn.metrics import r2_score

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_ = regressor.predict(test_x)
print('Mean absolute error: %.2f' % np.mean(np.absolute(test_y_ - test_y)))
print('Residual sum of squares (MSE): %.2f' % np.mean((test_y_ - test_y) **2 ))
print('R2-score: %.2f' % r2_score(test_y_, test_y))



