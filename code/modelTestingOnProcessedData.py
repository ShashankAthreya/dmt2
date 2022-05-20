# %%
# Importing libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
# %%
# Defining datafolder and importing dataframe
datafile = '../Dataset/VU_DM_data/updated_training_set_VU_DM.csv'
df_train = pd.read_csv(datafile)
# %%
# Declare feature vector and target variable and finding clickbool
import re
df_train = df_train.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
X_click = df_train.drop(['date_time','gross_bookings_usd', 'click_bool', 'position', 'booking_bool'], axis = 1)
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
# Declare feature vector and target variable and finding clickbool
X_book = df_train.drop(['date_time','gross_bookings_usd', 'position', 'booking_bool'], axis = 1)
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
# %%