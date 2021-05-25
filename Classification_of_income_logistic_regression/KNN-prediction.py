# -*- coding: utf-8 -*-
"""
Created on Tue May 25 15:20:03 2021

@author: vipin
"""

import os
os.chdir("path")
import pandas as pd
import seaborn as sns
import  numpy as np

#To partition the data

from sklearn.model_selection import train_test_split

#importing the library of KNN

from sklearn.neighbors import KNeighborsClassifier
#importing performace metrics-accuarcy score and confussion matrix

from sklearn.metrics import accuracy_score,confusion_matrix


#importing library for plotting

import matplotlib.pyplot as plt
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

#droping down insignificant values of 'gender','nativecountry','race','JobType'

cols=['gender','nativecountry','race','JobType']

new_data= data2.drop(cols,axis=1)

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

KNN_classifier=KNeighborsClassifier(n_neighbors=5)

#fitting values x and y

KNN_classifier.fit(train_x,train_y)

#predicting the test values

prediction=KNN_classifier.predict(test_x)

#performance matrics check

print(prediction)

#confusuion matrix
confusion_matrix=confusion_matrix(test_y,prediction)
print(confusion_matrix)

#calculating the accuracy
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
# Accuracy0.8396914738378153
#printing misclassified values from prediction

print('Misclassified samples : %d' % (test_y != prediction).sum())

#Misclassified samples : 1538
#calculating error for k values between 1 and 20
Misclassified_sample=[]
for i in range(1,20):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(train_x,train_y)
    pred_i=knn.predict(test_x)
    Misclassified_sample.append((test_y != pred_i).sum())
print(Misclassified_sample)

#missclassified samples 1761, 1499, 1606, 1501, 1538, 1469, 1481, 1443,
# 1466, 1439, 1450, 1444, 1445, 1434, 1454, 1418, 1424, 1418, 1433

#n_neighbors=17 has lowest misclassified samples