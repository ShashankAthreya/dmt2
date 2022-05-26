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
df_train = pd.read_csv(train_datafile, index_col=0)
test_datafile = '../Dataset/VU_DM_data/fixed_test_set_VU_DM.csv'
df_test = pd.read_csv(test_datafile, index_col=0)
print(df_train.columns, df_test.columns)
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
# Taking rows from the models
print(df_noclick.shape)
print(df_clickNoBook.shape) 
print(df_book.shape)
df_train = df_noclick.sample(n = 1300000)
df_train = df_train.append(df_clickNoBook, ignore_index=True)
df_train = df_train.append(df_book, ignore_index= True)
df_train = df_train.sort_values(by = 'srch_id', ascending=True)
# %%
# Declare feature vector and target variable and finding click_bool
df_train = df_train.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
X_click = df_train.drop(['click_bool', 'position', 'booking_bool'], axis = 1)
y_click = df_train['click_bool']

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
# Declare feature vector and target variable and finding booking_bool
X_book = df_train.drop(['position', 'booking_bool'], axis = 1)
y_book = df_train['booking_bool']

X_book_train, X_book_test, y_book_train, y_book_test = train_test_split(X_book, y_book, test_size = 0.3, random_state = 0)

clf_book = lgb.LGBMClassifier()
clf_book.fit(X_book_train, y_book_train)
# %% 
# Model Accuracy and predictions
y_book_pred=clf_book.predict(X_book_test)
accuracy=accuracy_score(y_book_pred, y_book_test)
print('LightGBM Model accuracy score: {0:0.4f}'.format(accuracy_score(y_book_test, y_book_pred)))

y_book_pred_train = clf_book.predict(X_book_train)
print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_book_train, y_book_pred_train)))

# Checking for overfitting
print('Training set score: {:.4f}'.format(clf_book.score(X_book_train, y_book_train)))
print('Test set score: {:.4f}'.format(clf_book.score(X_book_test, y_book_test)))

# Print the Confusion Matrix and slice it into four pieces
cm = confusion_matrix(y_book_test, y_book_pred)
print('Confusion matrix\n\n', cm)
print('\nTrue Positives(TP) = ', cm[0,0])
print('\nTrue Negatives(TN) = ', cm[1,1])
print('\nFalse Positives(FP) = ', cm[0,1])
print('\nFalse Negatives(FN) = ', cm[1,0])
# %
# %%
# Predicting on Test data
count = 0
df_test = df_test.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
X_click_test = df_test
df_test['click_bool'] = clf_click.predict(X_click_test)
X_book_test = df_test
df_test['booking_bool'] = clf_book.predict(X_book_test)
# %%
# Writing dataframe into a csv so I don't have to do this again.
df_test.to_csv("../Dataset/VU_DM_data/new_filled_test_set_VU_DM.csv")
# %%
print(df_test.loc[(df_test['click_bool'] == 0) & (df_test['booking_bool'] == 0)])
# %%
print(df_test.loc[(df_test['click_bool'] == 0) & (df_test['booking_bool'] == 1)])
# %%
print(df_test.loc[(df_test['click_bool'] == 1) & (df_test['booking_bool'] == 0)])
# %%
print(df_test.loc[(df_test['click_bool'] == 1) & (df_test['booking_bool'] == 1)])
# %%
