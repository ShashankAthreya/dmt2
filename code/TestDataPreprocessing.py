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
# Defining folder paths and dataframes for testing data
testData = '../Dataset/VU_DM_data/test_set_VU_DM.csv'
df_test = pd.read_csv(testData)
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
        rel_comp_inv.append(2.0)
        nan_count += 1
        continue
    score = 0
    for item in updatedRow:
        if item == 'nan':
            continue
        else:
            score += int(float(item))
    rel_comp_inv.append(float(score/competitors))
print(nan_count, rel_comp_inv.count(2.0))
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
        avg_comp_rate_diff.append(100.0)
        nan_count += 1
        continue
    score = 0
    for item in updatedRow:
        if item == 'nan':
            continue
        else:
            score += int(float(item))
    avg_comp_rate_diff.append(float(score/competitors)*total_comp_rate[index])
print(nan_count, avg_comp_rate_diff.count(100.0))
df_comp['avg_comp_rate_diff'] = avg_comp_rate_diff
# print(df_comp)
# %%
df_updated = df_test.drop(['comp1_rate','comp1_inv','comp1_rate_percent_diff','comp2_rate','comp2_inv','comp2_rate_percent_diff','comp3_rate','comp3_inv','comp3_rate_percent_diff','comp4_rate','comp4_inv','comp4_rate_percent_diff','comp5_rate','comp5_inv','comp5_rate_percent_diff','comp6_rate','comp6_inv','comp6_rate_percent_diff','comp7_rate','comp7_inv','comp7_rate_percent_diff','comp8_rate','comp8_inv','comp8_rate_percent_diff'], axis = 1)
df = pd.concat([df_updated, df_comp], axis = 1)
df.isnull().sum()
df_test = df.replace(np.nan, -1.0)
# %%
# Converting Datetime into seasons and time of day.
datetime = list(df_test['date_time'])
season = []
time_of_day = []
for entry in datetime:
    date = int(entry[5:7])*100 + int(entry[8:10])
    time = int(entry[11:13])*100 + int(entry[14:16])
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
df_test['season'] = season
df_test['time_of_day'] = time_of_day
# %%
# Writing dataframe into a csv so I don't have to do this again.
df_test.to_csv("../Dataset/VU_DM_data/updated_test_set_VU_DM.csv")
# %%
