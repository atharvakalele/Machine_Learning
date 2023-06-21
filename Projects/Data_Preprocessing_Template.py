# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 23:48:15 2023

@author: athar
"""

import pandas as pd
import numpy as np

path = r"C:\Users\athar\OneDrive\Desktop\Machine learning\Datasets\Machine Learning A-Z (Codes and Datasets)\Part 1 - Data Preprocessing\Section 2 -------------------- Part 1 - Data Preprocessing --------------------\Python\Data.csv"

data = pd.read_csv(path)

X = data.iloc[:, :-1].values
Y = data.iloc[:, -1].values

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])


from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_x = LabelEncoder()
X[:, 0] = labelencoder_x.fit_transform(X[:, 0])

onehotencoder = OneHotEncoder(sparse=False)
column_0_encoded = onehotencoder.fit_transform(X[:, 0].reshape(-1, 1))
print(column_0_encoded)
X = np.concatenate((column_0_encoded, X[:, 1:]), axis=1)
labelencoder_y = LabelEncoder()
Y = labelencoder_y.fit_transform(Y)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state = 0)