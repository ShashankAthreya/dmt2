# %%

# importing libraries
import datetime
from argon2 import DEFAULT_TIME_COST
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
# testData = r'C:\Users\Kikuk\source\repos\my-app\dmt2\data\test_set_VU_DM.csv'
# trainData = r'C:\Users\Kikuk\source\repos\my-app\dmt2\data\training_set_VU_DM.csv'
# %%

# Less data..
testData = r'C:\Users\Kikuk\source\repos\my-app\dmt2\data\500k_csv_testing.csv'
trainData = r'C:\Users\Kikuk\source\repos\my-app\dmt2\data\500k_csv_training.csv'

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
# test_headers = list(df_test.columns.values)
# train_headers = list(df_train.columns.values)
# %%

# Checking headers that need to be computed
# print(set(train_headers)-set(test_headers))

# %%

# Initial operation on training dataset
# df_train.head() # Checks for the headers of the df
# df_train.shape # Describes the size of the df
# df_train.dtypes # Describes the data_types used in the df
# df_train.nunique()  # Checks for Unique Values in each column of df
# df_train.isnull().sum() # Check for NULL values in each column
# df_train[df_train.isnull().any(axis=1)] # See rows with missing values
# df_train.describe() # Viewing the data statistics

# %%

# Finding out the correlation between the features
# corr = df_train.corr()
# corr.shape
# plt.figure(figsize=(100, 100))
# sns.heatmap(corr, cbar=True, square=True, fmt='.1f',
#             annot=True, annot_kws={'size': 35}, cmap='Greens')

# %%

# keep index name
df_train.index.name = 'id'
# ////////////// ID IS INDEX ACCORDING TO THE INITIAL READ CSV
# %%

# Combining srch_adult_cnt and srch_children_cnt columns
df_train['search_people_count'] = df_train['srch_adults_count'] + \
    df_train['srch_children_count']
# Dropping unecessary columns
df_train = df_train.drop(['srch_adults_count', 'srch_children_count', 'visitor_hist_starrating', 'prop_log_historical_price',
                         'srch_destination_id',  'srch_booking_window', 'srch_query_affinity_score', 'random_bool', 'gross_bookings_usd'], axis=1)


# %%

# Replacing NaNs with zeroes
df_newTrain = df_train.replace(np.nan, 0)
df_newTrain.head()
# %%

# Keeping only those who have clicked
df_has_booked = df_newTrain.loc[(df_newTrain['click_bool'] == 1)]
df_has_booked.head()
# %%

# converting the date_time column into datetime /// NECESSARY for the next step
df_has_booked['date_time'] = pd.to_datetime(df_has_booked['date_time'])
# %%

# list of months for ease of use later
months = []
for i in range(1, 13):
    months.append((i, datetime.date(1900, i, 1).strftime('%B')))


#  create a new df with only date_time
df_date_time = df_has_booked[['date_time']]

# saving the datetime and id of season-values
summer = df_date_time.loc[((df_date_time['date_time'].dt.month > months[4][0]) & (
    df_date_time['date_time'].dt.month < months[8][0]))]
fall = df_date_time.loc[((df_date_time['date_time'].dt.month > months[7][0]) & (
    df_date_time['date_time'].dt.month < months[11][0]))]
spring = df_date_time.loc[((df_date_time['date_time'].dt.month > months[1][0]) & (
    df_date_time['date_time'].dt.month < months[5][0]))]
winter = df_date_time.loc[((df_date_time['date_time'].dt.month == months[11][0]) | (
    df_date_time['date_time'].dt.month == months[0][0]) | (df_date_time['date_time'].dt.month == months[1][0]))]

# %%

# Getting the ids of the seasons-variables in order to get the correct indeces
# to use when adding season to the correct rows in the original df
summer_id = summer.index.to_list()
spring_id = spring.index.to_list()
fall_id = fall.index.to_list()
winter_id = winter.index.to_list()

# %%

# Creating a seasons dataframe with all the indeces
df_summer = {'season': 'summer', 'season_id': summer_id,
             'date_time': summer['date_time']}
df_summer = pd.DataFrame(df_summer)

df_winter = {'season': 'winter', 'season_id': winter_id,
             'date_time': winter['date_time']}
df_winter = pd.DataFrame(df_winter)

df_fall = {'season': 'fall', 'season_id': fall_id,
           'date_time': fall['date_time']}
df_fall = pd.DataFrame(df_fall)

df_spring = {'season': 'spring', 'season_id': spring_id,
             'date_time': spring['date_time']}
df_spring = pd.DataFrame(df_spring)

seasons = df_summer.append(df_winter).append(df_fall).append(df_spring)
# %%

# Adding correct seasons into df_has_booked

# list of df_has_booked indeces
df_has_booked_id_list = df_has_booked.index

# adding a season column to the original dataframe
df_has_booked['season'] = pd.DataFrame(columns=['season'])

for index in df_has_booked_id_list:
    season = seasons.loc[(seasons['season_id'] == index)]
    df_has_booked.at[index, 'season'] = season.at[index, 'season']

# df_has_booked.head(5)
# %%


# %%

# Getting the time of day
time = df_date_time['date_time'].dt.hour

# times of day
morning = time.loc[(time >= 6) & (time < 12)]
midday = time.loc[(time >= 12) & (time < 18)]
evening = time.loc[(time >= 18) & (time <= 23)]
night = time.loc[(time >= 0) & (time < 6)]

# ids of times of day
morning_id = morning.index.to_list()
midday_id = midday.index.to_list()
evening_id = evening.index.to_list()
night_id = night.index.to_list()

# Creating a time_of_day dataframe with all the indeces
df_morning = {'time_of_day': 'morning', 'time_of_day_id': morning_id}
df_morning = pd.DataFrame(df_morning)

df_midday = {'time_of_day': 'midday', 'time_of_day_id': midday_id, }
df_midday = pd.DataFrame(df_midday)

df_evening = {'time_of_day': 'evening', 'time_of_day_id': evening_id, }
df_evening = pd.DataFrame(df_evening)

df_night = {'time_of_day': 'night', 'time_of_day_id': night_id}
df_night = pd.DataFrame(df_night)

time_of_days = pd.concat(
    [df_morning, df_midday, df_evening, df_night], ignore_index=True)

# adding a time of day column to the original dataframe
df_has_booked['time_of_day'] = pd.DataFrame(columns=['time_of_day'])

# Adding time of day values to the column in df_has_booked
for index in range(len(time_of_days)):
    df_has_booked_id = time_of_days.at[index, 'time_of_day_id']
    time_of_day = time_of_days.at[index, 'time_of_day']
    df_has_booked.at[df_has_booked_id, 'time_of_day'] = time_of_day

# %%

# Time to drop some columns
df_has_booked.drop(['date_time', ], axis=1)

# Try to ... keep those who clicked, even if they didnt book a hotel
# %%

# Modeling time! //////////

# srch_id1 = df_newTrain.loc[(df_newTrain['srch_id'] == 839)]
# srch_id1.head(30)
# %%

# Finding out the correlation between the features
corr = df_has_booked.corr()['click_bool'].sort_values()
corr.shape
plt.figure(figsize=(100, 100))
sns.heatmap(corr, cbar=True, square=True, fmt='.1f',
            annot=True, annot_kws={'size': 35}, cmap='Greens')
# %%

# %%
