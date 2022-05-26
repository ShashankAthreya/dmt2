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
# %%
# Defining functions
# Function to calculate range of values in a list
def listRange(thisList):
    print(np.nanmin(thisList),np.nanmax(thisList))
    return
# %%
# %%
# Defining folder paths and dataframes for training data
# trainData = '../data/500k_csv_data.csv' 
trainData = '../Dataset/VU_DM_data/training_set_VU_DM.csv'
df_train = pd.read_csv(trainData)

df_comp1 = df_train[['srch_id', 'prop_id', 'comp1_rate', 'comp1_inv', 'comp1_rate_percent_diff']] 
df_comp2 = df_train[['srch_id', 'prop_id', 'comp2_rate', 'comp2_inv', 'comp2_rate_percent_diff']]
df_comp3 = df_train[['srch_id', 'prop_id', 'comp3_rate', 'comp3_inv', 'comp3_rate_percent_diff']] 
df_comp4 = df_train[['srch_id', 'prop_id', 'comp4_rate', 'comp4_inv', 'comp4_rate_percent_diff']] 
df_comp5 = df_train[['srch_id', 'prop_id', 'comp5_rate', 'comp5_inv', 'comp5_rate_percent_diff']] 
df_comp6 = df_train[['srch_id', 'prop_id', 'comp6_rate', 'comp6_inv', 'comp6_rate_percent_diff']] 
df_comp7 = df_train[['srch_id', 'prop_id', 'comp7_rate', 'comp7_inv', 'comp7_rate_percent_diff']] 
df_comp8 = df_train[['srch_id', 'prop_id', 'comp8_rate', 'comp8_inv', 'comp8_rate_percent_diff']] 
# %%
listRange(list(df_comp1['comp1_rate_percent_diff']))
listRange(list(df_comp2['comp2_rate_percent_diff']))
listRange(list(df_comp3['comp3_rate_percent_diff']))
listRange(list(df_comp4['comp4_rate_percent_diff']))
listRange(list(df_comp5['comp5_rate_percent_diff']))
listRange(list(df_comp6['comp6_rate_percent_diff']))
listRange(list(df_comp7['comp7_rate_percent_diff']))
listRange(list(df_comp8['comp8_rate_percent_diff']))
# %%
