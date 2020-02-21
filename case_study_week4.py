# =============================================================================================================================
# CLASSIFYING PERSONAL INCOME
# =============================================================================================================================
######################Required packages########################################################################################

# To work with  dataframes
import pandas as pd

# To perform munerical operations
import numpy as np

# To visualize data

import seaborn as sns
"""
import os
os.listdir()
os.chdir('Desktop/NPTEL')
os.listdir()
"""
# To partition the data
from sklearn.model_selection import train_test_split

# Importing library for logistic regression
from sklearn.linear_model import LogisticRegression


# Importing performance metrics - accuracy score and comfusion matrix
from sklearn.metrics import accuracy_score, confusion_matrix

# ============================================================
# Importing data 
# ============================================================
data_income = pd.read_csv('income.csv')

# Creating a copy of original data
data = data_income.copy()

"""
#Exploratory data analysis:

#1. Getting to know the data
#2. Data preprocessing (Missing values)
#3. Cross tables and data visualization

"""
# Getting to know the data
# *** To check variables data type

print(data.info())

#**** Check for missing values

data.isnull()

print('Data columns with null values:\n', data.isnull().sum())

#*** No missinf values !

#**** Summary of numerical variables

summary_num= data.describe()
print(summary_num)

#**** Summary of categorical variables 
summary_cate=data.describe(include="O")
print(summary_cate)

#**** Freqeuncy of each categories
data['JobType'].value_counts()
data['occupation'].value_counts()

#***Checking for unique classes
print(np.unique(data['JobType']))
print(np.unique(data['occupation']))
#*** There exists ' ?' instead of nan values
"""
Go back and read the data including "na_values[' ?']" to 
"""

data= pd.read_csv('income.csv', na_values=[" ?"])
data.isnull().sum()

# Data pre-processing

data.isnull().sum()

missing=data[data.isnull().any(axis=1)]
missing
#axis=1 => to consider at least one column value is missing

""" Points to note:
1. Missing values in Jobtype = 1809
2. Missing values in Occupation = 1816
3. There are 1809 rows where two specific columns i.e. occupation & Jobtype have missing values
4. (1816-1809) = 7 => You still have occupation unfilled for these 7 rows. Because, jobtype is Never worked 
"""
data2=data.dropna(axis=0)


# Relationship between independent variables
correlation=data2.corr()

# Cross tables and Data visualization
# Extracting the column names

data2.columns

gender = pd.crosstab(index=data2["gender"], columns='count', normalize=True)
print(gender)

# Gender vs Salary Status:

gender_salstat = pd.crosstab(index=data2["gender"],columns=data2['SalStat'],margins=True, normalize='index')
print(gender_salstat)


# Frequency distribution of 'Salary Status'
Salstat= sns.countplot(data2['SalStat'])

# Histogram of  Age
sns.distplot(data2['age'], bins=10, kde=False)
# People with age 20-45 age are high in frequency

# Box plot-Age vs Salary Status
sns.boxplot('SalStat','age', data=data2)
data2.groupby('SalStat')['age'].median()

# People with 35-50 age are more likely to earn > 50000 USD p.a.

## LOGISTICS REGRESSION

# Reindexing the salary status names to 0,1
data2['SalStat']=data2['SalStat'].map({' less than or equal to 50,000':0,'greater or equal to 50,0000':1})
print(data2['SalStat'])

new_data= pd.get_dummies(data2,drop_first=True)

# Storing the colimn names
columns_list=list(new_data.columns)
print(columns_list)

# Separating the input names from data

features = list(set(columns_list)-set(['SalStat']))
print(features)

# Storing the output values in y
y=new_data['SalStat'].values
print(y)

# storing the values from input features'x=new_data[features]