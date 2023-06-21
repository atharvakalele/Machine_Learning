# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 11:26:12 2023

@author: athar
"""
import pandas as pd

data = [[50, True], [40, False], [30, False]]

df = pd.DataFrame(data)
"""
print(df)
x = df.iloc[1, 0]

print(x)
df.iloc[1, 0] = 100
print(df)


print(df.iloc[[0,0,0,0,0,0,1]])
print(df.iloc[[0,0],[1,1,1]])
print(df.iloc[0:1])
print(df.values)
"""
data = {'age' : [12,18],
        'virgin' : ["No","YES"]
        }
df = pd.DataFrame(data)
print(df)
print(df.values)