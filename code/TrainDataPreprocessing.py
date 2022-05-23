# %%
# importing libraries
from cmath import nan
from curses.ascii import isdigit
import pandas as pd
import math
import numpy as np
from pyrsistent import v
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm
# %%
# Defining functions
# Function to calculate range of values in a list
def listRange(thisList):
    print(np.nanmin(thisList),np.nanmax(thisList))
    return

# Funtion to find correlation within a dataframe
def correlation(dataframe):
    corr = dataframe.corr()
    corr.shape
    plt.figure(figsize=(100, 100))
    sns.heatmap(corr, cbar=True, square=True, fmt='.1f',
                annot=True, annot_kws={'size': 35}, cmap='Greens')
    return
# %%
# Defining folder paths and dataframes for training data
# trainData = '../data/500k_csv_data.csv' 
trainData = '../Dataset/VU_DM_data/training_set_VU_DM.csv'
df_train = pd.read_csv(trainData)
# %%
# Defining folder paths and dataframes for testing data
testData = '../Dataset/VU_DM_data/test_set_VU_DM.csv'
df_test = pd.read_csv(testData)
# %%
# Making a list of headers
# train_headers = list(df_train.columns.values)
# test_headers = list(df_test.columns.values)
# print(set(train_headers)-set(test_headers))
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
# correlation(df_train)
# %%
# Creating a competitor dataframe
df_comp_rate = df_train[['comp1_rate','comp2_rate','comp3_rate','comp4_rate','comp5_rate','comp6_rate','comp7_rate','comp8_rate']]
df_comp_inv = df_train[['comp1_inv','comp2_inv','comp3_inv','comp4_inv','comp5_inv','comp6_inv','comp7_inv','comp8_inv']]
df_comp_rate_diff = df_train[['comp1_rate_percent_diff','comp2_rate_percent_diff','comp3_rate_percent_diff','comp4_rate_percent_diff','comp5_rate_percent_diff','comp6_rate_percent_diff','comp7_rate_percent_diff','comp8_rate_percent_diff']]
# %%
# Counting number of competitors and their rate
total_comp = []
total_comp_rate = []
nan_count = 0

# Does not yet work. Not loosing sleeping trying to make it work.
# def calcCompetition(dataframe):


# def mergingCompetitors(dataframe, title, nanReplacementValue):
#     nan_count = 0
#     for index,row in dataframe.iterrows():
#         updateableList = []
#         thisRow = list(dataframe.loc[[index][0]])
#         updatedRow = [str(x) for x in thisRow]
#         try:
#             competitors = total_comp[index]
#         except:
#             competitors = 8 - updatedRow.count('nan')
#             total_comp.append(competitors)
#         if competitors == 0:
#             updateableList.append(nanReplacementValue)
#             nan_count += 1
#             continue
#         score = 0
#         for item in updatedRow:
#             if item == 'nan':
#                 continue
#             else:
#                 score += int(float(item))
#         updateableList.append(float(score/competitors))
#     try:
#         df_comp[title] = updateableList
#     except:
#         df_comp = pd.DataFrame(total_comp, columns=['total_comp'])
#         df_comp[title] = updateableList
#     print(nan_count, updateableList.count(nanReplacementValue))
#     return

# mergingCompetitors(df_comp_rate,"total_comp_rate", 2.0)
for index,row in df_comp_rate.iterrows():
    thisRow = list(df_comp_rate.loc[[index][0]])
    updatedRow = [str(x) for x in thisRow]
    competitors = 8 - updatedRow.count('nan')
    total_comp.append(competitors)
    if competitors == 0:
        total_comp_rate.append(2.0)
        nan_count += 1
        continue
    score = 0
    for item in updatedRow:
        if item == 'nan':
            continue
        else:
            score += int(float(item))
    total_comp_rate.append(float(score/competitors))
print(nan_count, total_comp_rate.count(2.0))
# %%
# Creating a new dataframe with only competitors
df_comp = pd.DataFrame(total_comp, columns=['total_comp'])
df_comp['total_comp_rate'] = total_comp_rate
# %%
# Creating an inventory comparison with competitors
rel_comp_inv = []
nan_count = 0
for index,row in df_comp_inv.iterrows():
    thisRow = list(df_comp_inv.loc[[index][0]])
    updatedRow = [str(x) for x in thisRow]
    competitors = total_comp[index]
    if competitors == 0:
        rel_comp_inv.append(10.0)
        nan_count += 1
        continue
    score = 0
    for item in updatedRow:
        if item == 'nan':
            continue
        else:
            score += int(float(item))
    rel_comp_inv.append(float(score/competitors))
df_comp['rel_comp_inv'] = rel_comp_inv
# print(df_comp)
# %% 
# Creating a percentage difference between inventory prices
avg_comp_rate_diff = []
nan_count = 0
for index,row in df_comp_rate_diff.iterrows():
    thisRow = list(df_comp_rate_diff.loc[[index][0]])
    updatedRow = [str(x) for x in thisRow]
    competitors = total_comp[index]
    if competitors == 0:
        avg_comp_rate_diff.append(800000.0)
        nan_count += 1
        continue
    score = 0
    for item in updatedRow:
        if item == 'nan':
            continue
        else:
            score += int(float(item))
    avg_comp_rate_diff.append(float(score/competitors)*total_comp_rate[index])
df_comp['avg_comp_rate_diff'] = avg_comp_rate_diff
# print(df_comp)
# %%
# Updating the dataframe and checking for NaNs
df_updated = df_train.drop(['comp1_rate','comp1_inv','comp1_rate_percent_diff','comp2_rate','comp2_inv','comp2_rate_percent_diff','comp3_rate','comp3_inv','comp3_rate_percent_diff','comp4_rate','comp4_inv','comp4_rate_percent_diff','comp5_rate','comp5_inv','comp5_rate_percent_diff','comp6_rate','comp6_inv','comp6_rate_percent_diff','comp7_rate','comp7_inv','comp7_rate_percent_diff','comp8_rate','comp8_inv','comp8_rate_percent_diff'], axis = 1)
df = pd.concat([df_updated, df_comp], axis = 1)
df.isnull().sum()
# %%
# Creating Lists for columns with missing values
# visitor_hist_starrating = list(df['visitor_hist_starrating'])
# visitor_hist_adr_usd = list(df['visitor_hist_adr_usd'])
# prop_review_score = list(df['prop_review_score'])
# prop_location_score2 = list(df['prop_location_score2'])
# srch_query_affinity_score = list(df['srch_query_affinity_score'])
# orig_destination_distance = list(df['orig_destination_distance'])
# gross_bookings_usd = list(df['gross_bookings_usd'])

# print("visitor_hist_adr_usd")
# listRange(visitor_hist_adr_usd)
# print("visitor_hist_starrating")
# listRange(visitor_hist_starrating)
# print("prop_review_score")
# listRange(prop_review_score)
# print("prop_location_score2")
# listRange(prop_location_score2)
# print("srch_query_affinity_score")
# listRange(srch_query_affinity_score)
# print("orig_destination_distance")
# listRange(orig_destination_distance)
# print("gross_bookings_usd")
# listRange(gross_bookings_usd)
# %%
# Replacing NaNs with -1
df_train = df.replace(np.nan, -1.0)
# %%

# print(nan_count, avg_comp_rate_diff.count(100.0))
# print(nan_count, total_comp_rate.count(2.0))
# print(nan_count, rel_comp_inv.count(2.0))
# listRange(total_comp)
# listRange(total_comp_rate)
# listRange(rel_comp_inv)
# listRange(avg_comp_rate_diff)
# %%
# Converting Datetime into seasons and time of day.
datetime = list(df_train['date_time'])
season = []
time_of_day = []
month = []
hour = []
for entry in datetime:
    date = int(entry[5:7])*100 + int(entry[8:10])
    month.append(int(entry[5:7]))
    time = int(entry[11:13])*100 + int(entry[14:16])
    hour.append(int(entry[11:13]))
    if date >= 320 and date <= 620:
        season.append(1)
    if date >= 621 and date <= 920:
        season.append(2)
    if date >= 921 and date <= 1220:
        season.append(3)
    if (date >= 1221 and date <= 1231) or (date >= 101 and date <= 319):
        season.append(4)
    if time >= 600 and time <= 1159:
        time_of_day.append(1)
    if time >= 1200 and time <= 1559:
        time_of_day.append(2)
    if time >= 1600 and time <= 1959:
        time_of_day.append(3)
    if (time >= 2000 and time <= 2359) or (time >= 0 and time <= 559):
        time_of_day.append(4)
df_train['season'] = season
df_train['time_of_day'] = time_of_day
df_train['month'] = month
df_train['hour'] = hour
# %%
# Fixing Headers for testing and training the model
df_train = df_train[['srch_id', 'site_id',
       'visitor_location_country_id', 'visitor_hist_starrating',
       'visitor_hist_adr_usd', 'prop_country_id', 'prop_id', 'prop_starrating',
       'prop_review_score', 'prop_brand_bool', 'prop_location_score1',
       'prop_location_score2', 'prop_log_historical_price',
       'price_usd', 'promotion_flag', 'srch_destination_id',
       'srch_length_of_stay', 'srch_booking_window', 'srch_adults_count',
       'srch_children_count', 'srch_room_count', 'srch_saturday_night_bool',
       'srch_query_affinity_score', 'orig_destination_distance', 'random_bool',
       'total_comp', 'total_comp_rate', 'rel_comp_inv', 'avg_comp_rate_diff', 'season',
       'time_of_day', 'month', 'hour', 'click_bool',  'booking_bool', 'position']]
# %%
# Writing dataframe into a csv so I don't have to do this again.
df_train.to_csv("../Dataset/VU_DM_data/fixed_training_set_VU_DM.csv")
# %%
# %%
