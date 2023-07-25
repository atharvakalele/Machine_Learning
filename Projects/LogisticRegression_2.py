# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 12:22:09 2023

@author: athar
"""
#Filter-Based Feature Selection

import pandas as pd
import numpy as np

df = pd.read_csv(r"C:\Users\athar\OneDrive\Desktop\Machine learning\Projects\LogisticRegressionDataset\test.csv").drop('subject',axis=1)

#562 columns....
X = df.iloc[:,:-1]
y = df.iloc[:,-1]

#remove duplicate features
def duplicates_filter(df):
    duplicate_features=[]
    seen_features=set()
    for col in df.columns:
        col_data=df[col].values
        if(tuple(col_data) in seen_features):
            duplicate_features.append(col)
        else:
            seen_features.add(tuple(col_data))
    return df.drop(columns=duplicate_features,axis=1)

X = duplicates_filter(X)

#variance threshold
from sklearn.feature_selection import VarianceThreshold
sel = VarianceThreshold(threshold=0.05)
sel.fit(X)
columns = X.columns[sel.get_support()]
X = sel.fit_transform(X)
X = pd.DataFrame(X,columns = columns)

#pearson correlation
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

df_new = pd.DataFrame(X)
df_new['Activity'] = y

def pearson_correlation_filter(df_new):
    corr_matrix = df_new.corr()
    """target = df_new.columns[-1]
    irrelevant_features=[]
    #irrelevant_features
    for x in df_new.columns[:-1]:
        cor = corr_matrix.loc[x,target]
        if(abs(cor)<0.70):
            irrelevant_features.append(x)
    df_new = df_new.drop(columns=irrelevant_features,axis=1)"""
    #multicollinearity
    irrelevant_features = set()
    for x in df_new.columns[:-1]:
        for y in df_new.columns[:-1]:
            if(x!=y and abs(corr_matrix.loc[x,y])>0.97):
                irrelevant_features.add(y)
    irrelevant_features = list(irrelevant_features)
    df_new = df_new.drop(columns=irrelevant_features,axis=1)
    X = df_new.iloc[:,:-1]
    return X
X = pearson_correlation_filter(df_new)
#Reapplying Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)
 
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)
acs = accuracy_score(y_test, y_pred)


    
        
    