# ======================================================
# CLASSIFYING PERSONAL INCOME
# ======================================================
######################Reuquired packages################
# To work with dataframes
import pandas as pd

# To perform numerical operations
import numpy as np

# TP visualize data
import seaborn as sns

# To partition the data
from sklearn.model_selection import train_test_split

# Importing library for logistic regression
from sklearn.linear_model import LogisticRegression

# Importing performance metrics-accuracy score and confusion matrix
from sklearn.metrics import accuracy_score,confusion_matrix

##########################################################
# ========================================================
# Importing Data
# ========================================================
"""
import os
os.listdir()
os.chdir('Desktop')
os.chdir('NPTEL')
"""
data_income=pd.read_csv('income.csv')

# Creating a copy of original data
data = data_income.copy()

# TO check variable data type
print(data.info())

# Check for missing values
data.isnull()
print('Data columns with null values:\n', data.isnull().sum())

# No missing values !

# Summary of numerical variables
summary_num= data.describe()
print(summary_num)

# Summary of numerical variables
summary_cate= data.describe(include="O")
print(summary_cate)

# Frequency of each categories
data['JobType'].value_counts()
data['occupation'].value_counts()
 
# Checking for uniques classes
print(np.unique(data['JobType']))
print(np.unique(data['occupation']))

# There is exists ' ?' instead of nan

# Go back and read the data by including "na+values" to check missing values

data = pd.read_csv('income.csv',na_values=[" ?"])

# Data Pre-Processing

data.isnull().sum()

missing=data[data.isnull().any(axis=1)]


data2=data.dropna(axis=0)
correlation=data2.corr()

# Gender proportion table:

gender =pd.crosstab(index=data2["gender"],columns='count',normalize=True)
print(gender)

# Gender vs salary status

gender_salstat=pd.crosstab(index=data2["gender"],columns=data['SalStat'],margins=True,normalize='index')
print(gender_salstat)

import seaborn as sns
# Frequency distribution of salary status
SalStat=sns.countplot(data2['SalStat'])

# Histogram of Age
sns.distplot(data2['age'], bins=10, kde=False)

# Box-plot age vs salary status
sns.boxplot('SalStat','age',data=data2)
data2.groupby('SalStat')['age'].median()

# LOGISTIC REGRESSION MODEL

# Reindexing the salary status names to 0,1

data2['SalStat']=data2['SalStat'].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])

new_data=pd.get_dummies(data2,drop_first=True)
print(new_data)

# Storing the columns name
columns_list=list(new_data.columns)
print(columns_list)

# Seperating the input names from data
features=list(set(columns_list)-set(['SalStat']))
print(features)

# Storing the values in y
y=new_data['SalStat'].values
print(y)

# Storing the values in from the input features
x=new_data[features].values
print(x)

# Splitting the data into train and test
train_x, test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
# test_size represents the proportion of the data set include in the test split 
# random_state if 0 same samples are choosen for test and if not 0 then it will randomaly choose the samples

# make the instance of the model
logistic= LogisticRegression()

# fitting the values  for the x and y
logistic.fit(train_x,train_y)
logistic.coef_
logistic.intercept_

# prediction from the test data
prediction=logistic.predict(test_x)
print(prediction)

# Confusion matrix
confusion_matrix=confusion_matrix(test_y,prediction)
print(confusion_matrix)

# Calculating the accuracy
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)

# printing the missiclassified values from prediction
print('missiclassified samples: %d' % (test_y !=prediction).sum())



# ================================================================
# Logistic Regression - Removing the insignificant variables
# ================================================================

data2['SalStat']=data2['SalStat'].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])

cols=['gender','nativecountry','race','JobType']

new_data=data2.drop(cols,axis=1)

new_data=pd.get_dummies(new_data,drop_first=True)

#Storing the column names

columns_list=list(new_data.columns)
print(columns_list)

# Seperating the input names from data

features=list(set(columns_list)-set(['SalStat']))
print(features)

# Storing the output values in y
y=new_data['SalStat'].values 
print(y)

# Storing the output values from input features
x=new_data[features].values
print(x)

# Splitting the data into train and test

train_x, test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)

# make the instance of the model
logistic= LogisticRegression()
# fitting the values  for the x and y
logistic.fit(train_x,train_y)
logistic.coef_
logistic.intercept_

# prediction from the test data
prediction=logistic.predict(test_x)
print(prediction)

# Confusion matrix
confusion_matrix=confusion_matrix(test_y,prediction)
print(confusion_matrix)

# Calculating the accuracy
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)

# printing the missiclassified values from prediction
print('missiclassified samples: %d' % (test_y !=prediction).sum())

#=================================================================
# KNN
#=================================================================

# Importing the library of KNN
from sklearn.neighbors import KNeighborsClassifier

# import library for plotting
import matplotlib.pyplot as plt

# Storing the K nearest neighbors classifier
KNN_classifier=KNeighborsClassifier(n_neighbors=10)

# Fitting the values for X and Y
KNN_classifier.fit(train_x,train_y)

# Predicting the test values with model
prediction=KNN_classifier.predict(test_x)

# performance metric check
confusion_matrix=confusion_matrix(test_y,prediction)
print("\t","Predicted values")
print("Original values","\n",confusion_matrix)

# Calculating the accuracy
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)

print('missiclassified samples: %d' % (test_y !=prediction).sum())


"""
Effect of K values on classifier
"""
Missclassified_sample=[]

# Calculating the error for K values between 1 and 20

for i in range(1,20):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(train_x,train_y)
    pred_i=knn.predict(test_x)
    Missclassified_sample.append((test_y!=pred_i).sum())
    
print(Missclassified_sample)