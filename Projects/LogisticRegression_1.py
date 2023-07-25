# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 12:17:14 2023

@author: athar
"""

import pandas as pd
import numpy as np

df = pd.read_csv(r"C:\Users\athar\OneDrive\Desktop\Machine learning\Projects\LogisticRegressionDataset\test.csv").drop('subject',axis=1)
X = df.iloc[:,:-1]
y = df.iloc[:,-1]

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
y = labelencoder.fit_transform(y)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42) 

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train,y_train)
y_pred = log_reg.predict(X_test)

acs = accuracy_score(y_test, y_pred)

#duplicates
def detect_and_remove_duplicates(df):
    duplicate_features = []
    seen_features = set()
    for col in df.columns:
        col_data = df[col].values
        if tuple(col_data) in seen_features:
            duplicate_features.append(col)
        else:
            seen_features.add(tuple(col_data))
    return df.drop(columns=duplicate_features)


X_train = detect_and_remove_duplicates(X_train)
X_test = detect_and_remove_duplicates(X_test)

#Variance Thresholding
from sklearn.feature_selection import VarianceThreshold

threshold_value = 0.05  
variance_thresholder = VarianceThreshold(threshold=threshold_value)

# Fit the VarianceThreshold object on X_train and transform X_train
X_train = variance_thresholder.fit_transform(X_train)

# Transform X_test based on the variance threshold from X_train
X_test = variance_thresholder.transform(X_test)
X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)

#Correlation
import seaborn as sns
corr_matrix = X_train.corr()
columns = corr_matrix.columns
columns_to_drop = []
for i in range(len(columns)):
    for j in range(i+1,len(columns)):
        if(corr_matrix.loc[columns[i],columns[j]]>0.95):
            columns_to_drop.append(columns[j])

columns_to_drop=set(columns_to_drop)
X_train.drop(columns = columns_to_drop,axis=1,inplace=True)
X_test.drop(columns = columns_to_drop,axis=1,inplace=True)

#ANOVA
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest

sel = SelectKBest(f_classif,k=100).fit(X_train,y_train)
columns = X_train.columns[sel.get_support()]
X_train = sel.transform(X_train)
X_test = sel.transform(X_test)
X_train = pd.DataFrame(X_train,columns=columns)
X_test = pd.DataFrame(X_test,columns=columns)

#Reapplying Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)
acs = accuracy_score(y_test, y_pred)

