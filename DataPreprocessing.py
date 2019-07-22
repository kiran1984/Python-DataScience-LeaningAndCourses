# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 14:06:09 2019

@author: Alexandru
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#preparing data
data = pd.read_csv("Data.csv")
X=data.loc[:,['Country','Age','Salary']]
y=data.Purchased

#preprocessing the data
numerical_cols=[col for col in X.columns if X[col].dtype in ['int64','float64']]
categorical_cols=[col for col in X.columns if X[col].dtype=='object']

#preprocessing the numerical data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN',strategy='mean')
imputer = imputer.fit(X.loc[:,numerical_cols])
X[numerical_cols]=imputer.transform(X[numerical_cols])

#preprocessing the categorical data
from sklearn.preprocessing import OneHotEncoder
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_X = pd.DataFrame(OH_encoder.fit_transform(X[categorical_cols]))
OH_cols_X.index = X.index
num_X_train = X.drop(categorical_cols, axis=1)
OH_X_train = pd.concat([num_X_train, OH_cols_x], axis=1)
X=OH_X_train

#label encoding on y variable
from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

#spliting X and y
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0,test_size=0.2)