# %%
# Importing libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import re
# %%
# Defining datafolder and importing dataframe
train_datafile = '../Dataset/VU_DM_data/fixed_training_set_VU_DM.csv'
df_train = pd.read_csv(train_datafile)
test_datafile = '../Dataset/VU_DM_data/fixed_test_set_VU_DM.csv'
df_test = pd.read_csv(test_datafile)
# print(df_train.columns, df_test.columns)
# %%
# Taking rows from the models
df_noclick = df_train.loc[df_train['rank'] == 0]
df_click = df_train.loc[df_train['rank'] == 1]
df_book = df_train.loc[df_train['rank'] == 2]
df_train = df_noclick.sample(n = 500000)
df_train = df_train.append(df_click, ignore_index=True)
df_train = df_train.append(df_book, ignore_index= True)
df_train = df_train.sort_values(by = 'query_id', ascending=True)
# %%
# Declare feature vector and target variable and finding click_bool
df_train = df_train.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
X_click = df_train.drop(['rank'], axis = 1)
y_click = df_train['rank']

# X_click = df_train[[
#                     'query_id', 'weekday', 'avg_comp_rate_diff', 'srch_booking_window',
#                     'prev_click', 'srch_length_of_stay', 'abs_gross_booking_usd',
#                     'srch_query_affinity_score', 'total_comp_rate', 'prop_location_score1',
#                     'orig_destination_distance', 'srch_children_count', 'rel_comp_inv', 'prop_review_score',
#                     ]]
# y_click = df_train['rank']

X_click_train, X_click_test, y_click_train, y_click_test = train_test_split(X_click, y_click, test_size = 0.3, random_state = 0)

clf_click = lgb.LGBMClassifier()
clf_click.fit(X_click_train, y_click_train)
# %% 
# Model Accuracy and predictions
y_click_pred=clf_click.predict(X_click_test)
accuracy=accuracy_score(y_click_pred, y_click_test)
print('LightGBM Model accuracy score: {0:0.4f}'.format(accuracy_score(y_click_test, y_click_pred)))

y_click_pred_train = clf_click.predict(X_click_train)
print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_click_train, y_click_pred_train)))

# Checking for overfitting
print('Training set score: {:.4f}'.format(clf_click.score(X_click_train, y_click_train)))
print('Test set score: {:.4f}'.format(clf_click.score(X_click_test, y_click_test)))

# Print the Confusion Matrix and slice it into four pieces
cm = confusion_matrix(y_click_test, y_click_pred)
print('Confusion matrix\n\n', cm)
print('\nTrue Positives(TP) = ', cm[0,0])
print('\nTrue Negatives(TN) = ', cm[1,1])
print('\nFalse Positives(FP) = ', cm[0,1])
print('\nFalse Negatives(FN) = ', cm[1,0])
# %%
# Predicting on Test data
count = 0
df_test = df_test.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
X_click_test = df_test
# X_click_test = df_test[[
                        # 'query_id', 'weekday', 'avg_comp_rate_diff', 'srch_booking_window',
                        # 'prev_click', 'srch_length_of_stay', 'abs_gross_booking_usd',
                        # 'srch_query_affinity_score', 'total_comp_rate', 'prop_location_score1',
                        # 'orig_destination_distance', 'srch_children_count', 'rel_comp_inv', 'prop_review_score',
                        # ]]
df_test['rank'] = clf_click.predict(X_click_test)
# %%
print(df_test.loc[(df_test['rank'] == 0)])
# %%
print(df_test.loc[(df_test['rank'] == 1)])
# %%
print(df_test.loc[(df_test['rank'] == 2)])
# %%
# Writing dataframe into a csv so I don't have to do this again.
df_test.to_csv("../Dataset/VU_DM_data/new_filled_model_test_set_VU_DM.csv", index=False)
# df_test.to_csv("../Dataset/VU_DM_data/feature_model_test_set_VU_DM.csv", index=False)