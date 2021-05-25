# -*- coding: utf-8 -*-
"""
Created on Tue May 25 15:12:30 2021

@author: vipin
"""

import os
os.chdir("path")
import pandas as pd
import seaborn as sns
import  numpy as np

#To partition the data

from sklearn.model_selection import train_test_split

#importing library for logestic regression

from sklearn.linear_model import LogisticRegression

#importing performace metrics-accuarcy score and confussion matrix

from sklearn.metrics import accuracy_score,confusion_matrix

#importing data

data_income=pd.read_csv('income.csv',skipinitialspace=True)

#creating copy of orginal data

data=data_income.copy()

#To check data types of each variables
print(data.info())

data2 = data.dropna(axis=0) 

#checking missing values

data2.isnull().sum()

#Reindexing the salary status names to 1 and 0
data2['SalStat']=data2['SalStat'].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])

#one hot encoding
new_data=pd.get_dummies(data2,drop_first=True)

#storing the colunm names
columns_list=list(new_data.columns)
print(columns_list)

#seperate the input names from data -excluding SalStat from list and store as features
features=list(set(columns_list)-set(['SalStat']))
print(features)

#storing the output values in y
y=new_data['SalStat'].values
print(y)

#storing values from input features

x=new_data[features].values
print(x)

#splitting the data into train and test

train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)

#make an instance of the model
logistic=LogisticRegression(solver='lbfgs',max_iter=10000)

#fitting the values for x and y

logistic.fit(train_x,train_y)
logistic.coef_
logistic.intercept_

#prediction from test data

prediction=logistic.predict(test_x)

print(prediction)

#confusuion matrix
confusion_matrix=confusion_matrix(test_y,prediction)
print(confusion_matrix)

#calculating the accuracy
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
# accuracy 0.8424135263565035
#printing misclassified values from prediction

print('Misclassified samples : %d' % (test_y != prediction).sum())
#Misclassified samples : 1426