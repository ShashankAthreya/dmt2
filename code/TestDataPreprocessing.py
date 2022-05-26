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
# Defining folder paths and dataframes for testing data
testData = '../Dataset/VU_DM_data/test_set_VU_DM.csv'
df_test = pd.read_csv(testData)
# %%
# Adding new columns for features
total_prop = list(df_test['prop_id'].unique())  
with open('../Dataset/VU_DM_data/abs_usd.txt') as f:
    gross_usd_data = f.read()
with open('../Dataset/VU_DM_data/abs_click.txt') as f:
    all_clicks = f.read()
with open('../Dataset/VU_DM_data/abs_book.txt') as f:
    all_books = f.read()
final_dict = ast.literal_eval(gross_usd_data)

complete_usd = []
props = list(df_test['prop_id'])
for prop in props:
    try:
        complete_usd.append(final_dict[prop])
    except:
        complete_usd.append(0)
        print(prop)
df_test['abs_gross_booking_usd'] = complete_usd

click_dict = ast.literal_eval(all_clicks)
book_dict = ast.literal_eval(all_books)
all_clicks = []
all_books = []
props = list(df_test['prop_id'])
for prop in props:
    try:
        all_clicks.append(click_dict[prop])
        all_books.append(book_dict[prop])
    except:
        all_clicks.append(0)
        all_books.append(0)
        print(prop)
df_test['prev_click'] = all_clicks
df_test['prev_books'] = all_books
# %%
# Creating a competitor dataframe
df_comp_rate = df_test[['comp1_rate','comp2_rate','comp3_rate','comp4_rate','comp5_rate','comp6_rate','comp7_rate','comp8_rate']]
df_comp_inv = df_test[['comp1_inv','comp2_inv','comp3_inv','comp4_inv','comp5_inv','comp6_inv','comp7_inv','comp8_inv']]
df_comp_rate_diff = df_test[['comp1_rate_percent_diff','comp2_rate_percent_diff','comp3_rate_percent_diff','comp4_rate_percent_diff','comp5_rate_percent_diff','comp6_rate_percent_diff','comp7_rate_percent_diff','comp8_rate_percent_diff']]
# %%
# Counting number of competitors and their rate
total_comp = []
total_comp_rate = []
nan_count = 0

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
df_updated = df_test.drop(['comp1_rate','comp1_inv','comp1_rate_percent_diff','comp2_rate','comp2_inv','comp2_rate_percent_diff','comp3_rate','comp3_inv','comp3_rate_percent_diff','comp4_rate','comp4_inv','comp4_rate_percent_diff','comp5_rate','comp5_inv','comp5_rate_percent_diff','comp6_rate','comp6_inv','comp6_rate_percent_diff','comp7_rate','comp7_inv','comp7_rate_percent_diff','comp8_rate','comp8_inv','comp8_rate_percent_diff'], axis = 1)
df = pd.concat([df_updated, df_comp], axis = 1)
df.isnull().sum()
df_test = df.replace(np.nan, -1.0)
# %%
# Converting Datetime into seasons and time of day.
df_test['weekday'] = pd.to_datetime(df_test['date_time'])
df_test['weekday'] = df_test['weekday'].dt.day_of_week
# %%
# Fixing Headers for testing and training the model
df_test = df_test[['srch_id', 'site_id', 'weekday',
       'visitor_location_country_id', 'visitor_hist_starrating',
       'visitor_hist_adr_usd', 'prop_country_id', 'prop_id', 'prop_starrating',
       'prop_review_score', 'prop_brand_bool', 'prop_location_score1',
       'prop_location_score2', 'prop_log_historical_price',
       'price_usd', 'promotion_flag', 'srch_destination_id',
       'srch_length_of_stay', 'srch_booking_window', 'srch_adults_count',
       'srch_children_count', 'srch_room_count', 'srch_saturday_night_bool',
       'srch_query_affinity_score', 'orig_destination_distance', 'random_bool',
       'total_comp', 'total_comp_rate', 'rel_comp_inv', 'avg_comp_rate_diff', 'abs_gross_booking_usd', 'prev_click', 'prev_books']]
# %%
# Writing dataframe into a csv so I don't have to do this again.
df_test.to_csv("../Dataset/VU_DM_data/fixed_test_set_VU_DM.csv", index=False)
# %%
