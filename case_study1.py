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

# Frequency distr