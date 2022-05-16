# %%

# importing libraries
import datetime
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
testData = r'C:\Users\Kikuk\source\repos\my-app\dmt2\data\test_set_VU_DM.csv'
trainData = r'C:\Users\Kikuk\source\repos\my-app\dmt2\data\training_set_VU_DM.csv'
# %%

# Less data..
less_test_data = r'C:\Users\Kikuk\source\repos\my-app\dmt2\data\500k_csv_testing.csv'
less_training_data = r'C:\Users\Kikuk\source\repos\my-app\dmt2\data\500k_csv_training.csv'

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

# Replacing NaNs with zeroes
df_newTrain = df_train.replace(np.nan, 0)
df_newTrain.head()
# %%

# Keeping only those who have made a booking
df_has_booked = df_newTrain.loc[(df_newTrain['booking_bool'] == 1)]
df_has_booked.head()
# %%

# Seprating date and time

df_has_booked['date_time'] = pd.to_datetime(df_has_booked['date_time'])
# df_has_booked['Dates'] = pd.to_datetime(df_has_booked['date_time']).dt.date
# df_has_booked['Time'] = pd.to_datetime(df_has_booked['date_time']).dt.time
# df_split_dt = df_has_booked.drop(columns=['date_time'], axis=1)

# df_has_booked.head()
# %%
# Creating seasons column
months = []
for i in range(1, 13):
    months.append((i, datetime.date(1900, i, 1).strftime('%B')))


df_date_time = df_has_booked[['date_time']]

summer = df_date_time.loc[((df_date_time['date_time'].dt.month > months[4][0]) & (
    df_date_time['date_time'].dt.month < months[8][0]))]
fall = df_date_time.loc[((df_date_time['date_time'].dt.month > months[8][0]) & (
    df_date_time['date_time'].dt.month < months[11][0]))]
spring = df_date_time.loc[((df_date_time['date_time'].dt.month > months[2][0]) & (
    df_date_time['date_time'].dt.month < months[5][0]))]
winter = df_date_time.loc[((df_date_time['date_time'].dt.month == months[10][0]) | (
    df_date_time['date_time'].dt.month == months[0][0]) | (df_date_time['date_time'].dt.month == months[1][0]))]

summer.head()
# %%%

# adding season as a column
df_date_time['date_time'].dt.month.head()
summer['date_time'].dt.month.head()

print(df_date_time[['date_time']][0])

# if(df_date_time[['date_time']][0] == summer[['date_time']][0]):
#     print('hei')

# %%
