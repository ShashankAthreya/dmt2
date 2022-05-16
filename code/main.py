# %%
# importing libraries
import pandas as pd
import math
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm
# %%
# Defining folder path
# testData = './Dataset/VU_DM_data/test_set_VU_DM.csv'
trainData = './data/500k_csv_data.csv'
# %%
# Creating Dataframes for trainig data
df_train = pd.read_csv(trainData)
# print(df_train)
# %%
# Creating Dataframes for testing
df_test = pd.read_csv(testData)
# print(df_test)
# %%
# Making a list of headers
test_headers = list(df_test.columns.values)
train_headers = list(df_train.columns.values)
# %%
# Checking headers that need to be computed
print(set(train_headers)-set(test_headers))
# %%
# Initial operation on training dataset
# df_train.head() # Checks for the headers of the df
# df_train.shape # Describes the size of the df
# df_train.dtypes # Describes the data_types used in the df
# df_train.nunique() # Checks for Unique Values in each column of df
# df_train.isnull().sum() # Check for NULL values in each column
# df_train[df_train.isnull().any(axis=1)] # See rows with missing values
# df_train.describe() # Viewing the data statistics
# %%
# Finding out the correlation between the features
corr = df_train.corr()
corr.shape
plt.figure(figsize=(100, 100))
sns.heatmap(corr, cbar=True, square=True, fmt='.1f',
            annot=True, annot_kws={'size': 35}, cmap='Greens')
# %%
# Spliting target variable and independent variables
df_newTrain = df_train.replace('NaN', 0)
# df_newTrain.dropna()
df_newTrain.isnull().sum()
X = df_newTrain.drop(['date_time', 'gross_bookings_usd',
                     'click_bool', 'position', 'booking_bool'], axis=1)
y = df_newTrain['click_bool']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=4)

# Create a Linear regressor and train the model using the training sets
lm = LinearRegression()
lm.fit(X_train, y_train)
lm.intercept_  # Value of y intercept
# %%
#
df_newTrain = df_train.replace('NaN', 0)
# df_newTrain.dropna()
df_newTrain.isnull().sum()
# %%
