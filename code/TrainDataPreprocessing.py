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
import json
# importing the module
import ast
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
print(df_train.columns)

# %%
# Making a list of headers
# train_headers = list(df_train.columns.values)
# test_headers = list(df_test.columns.values)
# print(set(train_headers)-set(test_headers))

# Conbining gross booking usd
# df_usd = df_train[['prop_id', 'gross_bookings_usd']]
# df_usd = df_usd.replace(np.nan, 0.0)
total_prop = list(df_train['prop_id'].unique())  
with open('../Dataset/VU_DM_data/abs_usd.txt') as f:
    gross_usd_data = f.read()
with open('../Dataset/VU_DM_data/abs_click.txt') as f:
    all_clicks = f.read()
with open('../Dataset/VU_DM_data/abs_book.txt') as f:
    all_books = f.read()
final_dict = ast.literal_eval(gross_usd_data)
# %%
# # Wrting file to dict
# try:
#     dict_file = open('../Dataset/VU_DM_data/abs_usd.txt', 'wt')
#     dict_file.write(str(final_dict))
#     dict_file.close()
  
# except:
#     print("Unable to write to file")

# Adding it to dataframe
complete_usd = []
props = list(df_train['prop_id'])
for prop in props:
    try:
        complete_usd.append(final_dict[prop])
    except:
        complete_usd.append(0)
        print(prop)
df_train['abs_gross_booking_usd'] = complete_usd
# %%
# Storing All clicks and booking details to generate score
df_click_book = df_train[['prop_id','click_bool', 'booking_bool']]
# click_dict = {}
# book_dict = {}
# for prop in total_prop:
#     click_dict[prop] = sum(df_click_book.loc[df_click_book['prop_id']==prop,'click_bool'].tolist())
#     book_dict[prop] = sum(df_click_book.loc[df_click_book['prop_id']==prop,'booking_bool'].tolist())
# try:
#     dict_file = open('../Dataset/VU_DM_data/abs_click.txt', 'wt')
#     dict_file.write(str(click_dict))
#     dict_file.close()
  
# except:
#     print("Unable to write to file")
# try:
#     dict_file = open('../Dataset/VU_DM_data/abs_book.txt', 'wt')
#     dict_file.write(str(book_dict))
#     dict_file.close()
  
# except:
#     print("Unable to write to file")
click_dict = ast.literal_eval(all_clicks)
book_dict = ast.literal_eval(all_books)
all_clicks = []
all_books = []
props = list(df_train['prop_id'])
for prop in props:
    try:
        all_clicks.append(click_dict[prop])
        all_books.append(book_dict[prop])
    except:
        all_clicks.append(0)
        all_books.append(0)
        print(prop)
df_train['prev_click'] = all_clicks
df_train['prev_books'] = all_books
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
df_train['weekday'] = pd.to_datetime(df_train['date_time'])
df_train['weekday'] = df_train['weekday'].dt.day_of_week
# %%
# Creating a stable dataframe for models
df_noclick = df_train.loc[df_train['click_bool'] == 0]
df_clickNoBook = df_train.loc[(df_train['click_bool'] == 1) & (df_train['booking_bool'] == 0)]
df_book = df_train.loc[df_train['booking_bool'] == 1]
# %%
# Basic Operations
df_train['rank'] = df_train['click_bool'] + df_train['booking_bool']
print(list(df_train['rank']).count(0), list(df_train['rank']).count(1), list(df_train['rank']).count(2))

# %%
# Fixing Headers for testing and training the model
df_train = df_train[['srch_id', 'site_id', 'weekday',
       'visitor_location_country_id', 'visitor_hist_starrating',
       'visitor_hist_adr_usd', 'prop_country_id', 'prop_id', 'prop_starrating',
       'prop_review_score', 'prop_brand_bool', 'prop_location_score1',
       'prop_location_score2', 'prop_log_historical_price',
       'price_usd', 'promotion_flag', 'srch_destination_id',
       'srch_length_of_stay', 'srch_booking_window', 'srch_adults_count',
       'srch_children_count', 'srch_room_count', 'srch_saturday_night_bool',
       'srch_query_affinity_score', 'orig_destination_distance', 'random_bool',
       'total_comp', 'total_comp_rate', 'rel_comp_inv', 'avg_comp_rate_diff', 'abs_gross_booking_usd', 'prev_click', 'prev_books','rank','position']]
# %%
# Writing dataframe into a csv so I don't have to do this again.
df_train.to_csv("../Dataset/VU_DM_data/fixed_training_set_VU_DM.csv", index=False)
# %%
# %%
