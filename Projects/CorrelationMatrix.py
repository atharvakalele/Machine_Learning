# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 23:14:17 2023

@author: athar
"""

import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt


path = r"C:\Users\athar\OneDrive\Desktop\Machine learning\Projects\SimpleLinearRegressionDataset\FuelConsumption.csv"

dataset = pd.read_csv(path)


correlation_matrix = dataset.corr()#type(cm) = dataframe
sn.heatmap(correlation_matrix,annot = True)

