# -*- coding: utf-8 -*-
"""
Created on Tue May 25 15:11:00 2021

@author: vipin
"""

# =============================================================================
# importing neccesary packages for dataframes,data visualization and/ 
# numerical operations
# =============================================================================
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

#check missing values

print("Data columns with missing values :",data.isnull().sum())
#**no missing values found

# checking summary of numerical variables

summary_num=data.describe()
print(summary_num)

# checking sum o categorical varaiables

summary_cate = data.describe(include='O')
print(summary_cate)

#checking frequency of each categories

data['JobType'].value_counts()
#find irregular data of count 1809 under question mark
data['occupation'].value_counts()
#find irregular data of count 1816 under question mark
data['EdType'].value_counts()
#no irregular data found
data['maritalstatus'].value_counts()
#no irreqular data found
data['relationship'].value_counts()
#no irreqular data found
data['race'].value_counts()
#no irreqular data found
data['gender'].value_counts()
#no irreqular data found
data['nativecountry'].value_counts()
#no irreqular data found
data['SalStat'].value_counts()
#no irreqular data found

#Checking for unique classes of Jobtype and occupation
print(np.unique(data['JobType']))

# found a special character ' ?'
print(np.unique(data['occupation']))

# found a special character ' ?'

#Go back and read the data again by including ' ?' as na values

data=pd.read_csv('income.csv',na_values=' ?')

#checking data of Jobtype and occupation

data['JobType'].value_counts()
#no irregular data found

data['occupation'].value_counts()
#no irregular data found


# **************************************************************

# Data pre-processing

# **************************************************************

#checking missing value

data.isnull().sum()

#sub-set missing values

missing = data[data.isnull().any(axis=1)]
#axis=1 to consider at least one column value is missing

#Dropping down missing values since it is large data set

data2 = data.dropna(axis=0) 

#checking missing values

data2.isnull().sum()
#No missing values

#checking relationship between indepent variables

correlation = data2.corr()

#correlation values nearer to zero..so no corelation between variables

#**************************************************************************
#Cross tables & Data visualization

#**************************************************************************
#extracting columns name
data2.columns

#column names: 'age', 'JobType', 'EdType', 'maritalstatus', 'occupation',
#       'relationship', 'race', 'gender', 'capitalgain', 'capitalloss',
#       'hoursperweek', 'nativecountry', 'SalStat'

#Gender proportion table

gender = pd.crosstab(index=data2['gender'],columns='count',normalize=True)

#gender vs salary status

gender_salstat = pd.crosstab(index=data2['gender'],columns=data2['SalStat'],
                             margins=True,normalize='index')

#Frequency distribution of salary status

Salstat=sns.countplot(data2['SalStat'])

#75 % of the people slary status is <=50000

###################### Histogram of the Age ############################

sns.displot(data2['age'],bins=10,kde=False)

#People with 20-45 are high in frequency

#################### Box plot-Age vs Salary status ####################
sns.boxplot('SalStat','age',data=data2)
data2.groupby('SalStat')['age'].median()