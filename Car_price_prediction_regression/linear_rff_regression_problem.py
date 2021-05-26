# -*- coding: utf-8 -*-
"""
Created on Tue May 25 19:18:01 2021

@author: vipin
"""

# =============================================================================
# Predicting price of pre-owned cars
# =============================================================================

import  pandas as pd
import numpy as np
import seaborn as sns
import os
os.chdir("D:\\Training videos\\nptel\\python for dat science\\data sets\\week 4")
# =============================================================================
# setting dimensions for plot
# =============================================================================

sns.set(rc={'figure.figsize':(11.7,8.27)})

# =============================================================================
# Reading csv file
# =============================================================================

cars_data=pd.read_csv("cars_sampled.csv")

# =============================================================================
# 
# creating copy
# =============================================================================

cars=cars_data.copy()


# =============================================================================
# structure of data
# =============================================================================

cars.info()

#vehicle type,gearbox,model and fuel type have missing values

# =============================================================================
# summmerizing data
# =============================================================================

cars.describe()
pd.set_option('display.float_format',lambda x: '%.3f' % x)
#function for set only 3 decimal point
cars.describe()

# =============================================================================
# Droping unwanted columns
# =============================================================================

col=['name','dateCrawled','dateCreated','postalCode','lastSeen']
cars=cars.drop(columns=col,axis=1)
#these columns are not necessary

# =============================================================================
# Removing duplicate records
# =============================================================================

cars.drop_duplicates(keep='first',inplace=True)

#470 duplicate records found

# =============================================================================
# Data cleaning
# =============================================================================

#No of missing values in each column
cars.isnull().sum()
#vehicle type,gearbox,model and fuel type have missing values


#variable year of registration

yearwise_count=cars['yearOfRegistration'].value_counts().sort_index()
sum(cars['yearOfRegistration'] > 2018)
#26 cars >2018
sum(cars['yearOfRegistration'] < 1950)
#38 cars <1950
sns.regplot(x='yearOfRegistration',y='price',scatter=True,fit_reg=False,data=cars)
#setting working range in between 1950 and 2018

#variable price
price_count=cars['price'].value_counts().sort_index()
cars['price'].describe()
sns.boxplot(y=cars['price'])
sum(cars['price'] >150000)
#34 cars above 150000
sum(cars['price'] < 100)
#1748 cars below 100

#setting working range in between 1000 and 150000

#variable power ps
power_count=cars['powerPS'].value_counts().sort_index
cars['powerPS'].describe()
sns.regplot(x='powerPS',y='price',scatter=True,fit_reg=False,data=cars)
sum(cars['powerPS'] > 500)
#115 car above 500
sum(cars['powerPS'] < 10)
#5565 below 10

# =============================================================================
# Working range of data
# =============================================================================

cars=cars[(cars.yearOfRegistration <= 2018) 
          & (cars.yearOfRegistration >= 1950) 
          & (cars.price >=100)
          & (cars.price <= 150000)
          & (cars.powerPS >= 10)
          & (cars.powerPS <= 500)]
# ~ 6700 records dropped

#simplify variable reduction
#combining year of registration and month of registration

cars['monthOfRegistration'] /= 12
#this will make month to age

#creating new variable age by adding year of registration and month of 
#registration

cars['Age']= (2018-cars['yearOfRegistration']) + cars['monthOfRegistration']
cars['Age'] = round(cars['Age'],2)
cars['Age'].describe()

#Dropping down year ofregistration

cars=cars.drop(columns=['yearOfRegistration','monthOfRegistration'],axis=1)

#visualizing parameters

#Age
sns.displot(cars['Age'])
sns.boxplot(y=cars['Age'])

#price

sns.displot(cars['price'])
sns.boxplot(y=cars['price'])

#powerps
sns.displot(cars['powerPS'])
sns.boxplot(y=cars['powerPS'])

#Visualizing parametersafternarrowing working range

sns.regplot(x='Age',y='price',scatter=True,fit_reg=False,data=cars)
#carspriced higher are newer
#with increasein age price decreases
#however it is found some carpriced higher with increased age

#powerPS VS price 
sns.regplot(x='powerPS',y='price',scatter=True,fit_reg=False,data=cars)

#variable seller

cars['seller'].value_counts()
pd.crosstab(cars['seller'],columns='count',normalize=True)
sns.countplot(x='seller',data=cars)
#Fewercars have 'commerical' - insignificant

#variable offertype
cars['offerType'].value_counts()
pd.crosstab(cars['offerType'],columns='count',normalize=True)
sns.countplot(x='offerType',data=cars)

# All cars in offer- insignificant

#variable abtest
cars['abtest'].value_counts()
pd.crosstab(cars['abtest'],columns='count',normalize=True)
sns.countplot(x='abtest',data=cars)
#Equally distibuted 
sns.boxplot(x='abtest',y='price',data=cars)
#Forevery price value there is almost 50-50 distribution
# Does not affect price - insignificant


#variable vehicle type

cars['vehicleType'].value_counts()
pd.crosstab(cars['vehicleType'],columns='count',normalize=True)
sns.countplot(x='vehicleType',data=cars)
sns.boxplot(x='vehicleType',y='price',data=cars)

#vehicle types effect the price

#variable gearbox
cars['gearbox'].value_counts()
pd.crosstab(cars['gearbox'],columns='count',normalize=True)
sns.countplot(x='gearbox',data=cars)
sns.boxplot(x='gearbox',y='price',data=cars)
#gearbox types effect the price

#variable model
cars['model'].value_counts()
pd.crosstab(cars['model'],columns='count',normalize=True)
sns.countplot(x='model',data=cars)
sns.boxplot(x='model',y='price',data=cars)
#considered in modelling

#variable kilometer
cars['kilometer'].value_counts().sort_index()
pd.crosstab(cars['kilometer'],columns='count',normalize=True)
sns.countplot(x='kilometer',data=cars)
cars['kilometer'].describe()
sns.distplot(cars['kilometer'],bins=8,kde=False)
sns.regplot(x='kilometer',y='price',scatter=True,fit_reg=False,data=cars)
#considered in modelling


#variable fueltype
cars['fuelType'].value_counts()
pd.crosstab(cars['fuelType'],columns='count',normalize=True)
sns.countplot(x='fuelType',data=cars)
sns.boxplot(x='fuelType',y='price',data=cars)

#considered for modelling

#variable brand

cars['brand'].value_counts()
pd.crosstab(cars['brand'],columns='count',normalize=True)
sns.countplot(x='brand',data=cars)
sns.boxplot(x='brand',y='price',data=cars)
#cars aredistributedover many brands
#considered for modelling

#variable notRepairedDamage
cars['notRepairedDamage'].value_counts()
pd.crosstab(cars['notRepairedDamage'],columns='count',normalize=True)
sns.countplot(x='notRepairedDamage',data=cars)
# AS expected,the cars that require the damage to repaired
# fall under lower price


# =============================================================================
# Removing insignificant Variables
# =============================================================================

col=['seller','offerType','abtest']
cars=cars.drop(columns=col,axis=1)
cars_copy=cars.copy()


# =============================================================================
# correlation
# =============================================================================

cars_select1=cars.select_dtypes(exclude=[object])
correlation=cars_select1.corr()
round(correlation,1)
cars_select1.corr().loc[:,'price'].abs().sort_values(ascending=False)[1:]


#linerarregression and random forest model on two sets ofdata
# 1. Data obtained by omitting rows with any missing value
# 2. Data obtained by inputting the missing values


# =============================================================================
# Omitting the missing values
# =============================================================================

cars_omit=cars.dropna(axis=0)
#converting categorical variables to dummy variables (one hot encoding)

cars_omit=pd.get_dummies(cars_omit,drop_first=True)

# =============================================================================
# Importing necessary Libraries
# =============================================================================

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


# =============================================================================
# Model Building with Omitted Data
# =============================================================================

#seperating input and outputfeatures
x1 = cars_omit.drop(['price'],axis='columns',inplace=False)
y1=cars_omit['price']

#plotting thevariable price

prices = pd.DataFrame({"1. Before ": y1 ,"2 .After ":np.log(y1)})
prices.hist()

#Transforming price as logerthimic value

y1=np.log(y1)

#splitting data into test and train
X_train,X_test,y_train,y_test = train_test_split(x1,y1,test_size=0.3,random_state=3)
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)

# =============================================================================
# Baseline model for omitted data
# =============================================================================

#here making a base model by using test data mean value
# this is to set a benchmark and to compare with our regression model

#finding the mean for test data value

base_pred = np.mean(y_test)
print(base_pred)
#base_predict 8.249615787653337

# Repeating same value till length of test data
base_pred = np.repeat(base_pred,len(y_test))

#finding the RMSE

base_root_mean_square_error = np.sqrt(mean_squared_error(y_test,base_pred))

print(base_root_mean_square_error)

# base_root_mean_square_error = 1.1274483657478247

# =============================================================================
# Linear regression with ommited data
# =============================================================================

#setting intercept as true
lgr=LinearRegression(fit_intercept=True)

#model

model_lin1=lgr.fit(X_train,y_train)

#predicting model on test set

cars_predictions_lin1=lgr.predict(X_test)

#computing MSE and RMSE

lin_mse1= mean_squared_error(y_test,cars_predictions_lin1)
lin_rmse1 = np.sqrt(lin_mse1)
print(lin_rmse1)

#lin_rmse =0.5455481266513815

#R squared value
r2_lin_test1=model_lin1.score(X_test,y_test)
r2_lin_train1= model_lin1.score(X_train,y_train)
print(r2_lin_test1,r2_lin_train1)
# 0.7658615091649266 0.7800936978183916

#regression diagnostics - residual plot anlysis

residuals1=y_test - cars_predictions_lin1

sns.regplot(x=cars_predictions_lin1,y=residuals1,scatter=True, fit_reg=False)

residuals1.describe()


# =============================================================================
# Random Forest with ommitted data
# =============================================================================

#model parameters
rf = RandomForestRegressor(n_estimators=100,max_features='auto',max_depth=100,
                           min_samples_split=10,min_samples_leaf=4,random_state=1)

#model

model_rf1=rf.fit(X_train,y_train)

#Predicting model on test set

cars_predictions_rf1 = rf.predict(X_test)

#Computing MSE and RMSE

rf_mse1=mean_squared_error(y_test,cars_predictions_rf1)
rf_rmse1 = np.sqrt(rf_mse1)
print(rf_rmse1)
#rmse1 0.4360736289370223

# R squared value
r2_rf_test1=model_rf1.score(X_test,y_test)
r2_rf_train1 = model_rf1.score(X_train,y_train)
print(r2_rf_test1,r2_rf_train1)

#value= 0.8504018147750623 0.9202494705146291

# =============================================================================
# Model Building with imputed data
# =============================================================================

cars_imputed = cars.apply(lambda x:x.fillna(x.median()) \
                          if x.dtype=='float'  else \
                              x.fillna(x.value_counts().index[0]))
cars_imputed.isnull().sum()

# Converting categoricalvariables to dummy variables

cars_imputed = pd.get_dummies(cars_imputed,drop_first=True)

# =============================================================================
# Model building with imputed data
# 
# =============================================================================

#seperatinginput and output features

x2=cars_imputed.drop(['price'],axis='columns',inplace=False)
y2 = cars_imputed['price']

#plotting the variable price
prices = pd.DataFrame({"1. Before :" :y2,"After: ":np.log(y2)})
prices.hist()

# Transforming price asa logarthmicvalue
y2 = np.log(y2)

# Splitting datainto test and train 

X_train1,X_test1,y_train1,y_test1 = train_test_split(x2,y2,test_size=0.3,random_state=3)

print(X_train1.shape,X_test1.shape,y_train1.shape,y_test1.shape)
